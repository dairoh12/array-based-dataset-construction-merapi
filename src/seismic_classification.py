#!/usr/bin/env python3
"""
=============================================================================
Seismic Event Classification Pipeline
4060_ML_2026 — Konversi dari Jupyter Notebook ke standalone Python script

Penggunaan (one-click full pipeline):
    python seismic_ml_pipeline.py --mode all \
        --data_dir /path/to/mseed_folder \
        --output_csv /path/to/output.csv \
        --model_output /path/to/models \
        --output_plots /path/to/plots

Atau jalankan per tahap:
    python seismic_ml_pipeline.py --mode extract --data_dir ... --output_csv ...
    python seismic_ml_pipeline.py --mode train   --output_csv ... --model_output ...
    python seismic_ml_pipeline.py --mode evaluate --output_csv ... --model_output ...
=============================================================================

CATATAN REKONSTRUKSI (dibaca sebelum lanjut)
---------------------------------------------
File ini ditulis ulang oleh Claude karena SSD berisi file asli terputus
di tengah sesi kerja. Sebagian besar isi file ini adalah salinan PERSIS
dari versi terakhir yang sudah diverifikasi sepanjang sesi (dibaca penuh
dan/atau diedit langsung). Beberapa fungsi plotting/report di TAHAP 4
(PREDIKSI) tidak sempat terbaca isinya secara penuh sebelum SSD terputus —
fungsi-fungsi itu ditulis ulang dari nol agar tetap fungsional (nama,
parameter, dan cara pemanggilan tetap sama), dan masing-masing diberi
banner komentar:

    # ⚠️ REKONSTRUKSI ⚠️

Cari banner itu untuk menemukan bagian yang perlu dibandingkan ulang
dengan kode asli begitu SSD sudah bisa diakses lagi.
"""

import argparse
import glob
import os
import sys
import warnings
import json
import re

warnings.filterwarnings("ignore")

# ===========================================================================
# IMPORTS
# ===========================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from collections import defaultdict
from scipy.stats import skew, kurtosis

# ObsPy
from obspy import read, Stream
from obspy.core.util.attribdict import AttribDict


# Scikit-learn
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, cross_val_score, learning_curve
)
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_fscore_support, accuracy_score,
    make_scorer, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA


# Imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# XGBoost
from xgboost import XGBClassifier


# ===========================================================================
# KOORDINAT STASIUN DEFAULT (Merapi Array)
# Dapat di-override via argumen --station_coords (JSON string atau path file)
# ===========================================================================
DEFAULT_STATION_COORDS = {
    "RE5DE": (-7.692254, 110.438530),
    "R6940": (-7.692179, 110.441112),
    "R265F": (-7.694289, 110.438976),
    "R7D17": (-7.693931, 110.441316),
    "R0279": (-7.691258, 110.440031)
}

# Stasiun referensi untuk ekstraksi fitur statistik/spektral SATU stasiun
# (Mean, Std, ..., SpectralEntropy — lihat extract_features_from_trace()).
# TIDAK dipakai lagi oleh FK Analysis/Beamforming (fk_grid_search() dan
# beamforming() memakai SEMUA stasiun yang tersedia relatif terhadap
# centroid array, tidak di-anchor ke satu stasiun referensi).
REFERENCE_STATION = "R0279"

# Konversi derajat lintang → km (rata-rata Bumi, ~111.32 km/derajat).
# Dipakai fk_grid_search() & beamforming() untuk mengubah koordinat stasiun
# (lat, lon) menjadi koordinat lokal East-North dalam km, supaya Slowness
# bersatuan detik/km (standar seismologi). Konversi derajat bujur → km ikut
# dikalikan cos(latitude) karena jarak per derajat bujur menyempit
# mendekati kutub (untuk Merapi ~ -7.69°, faktor ini ~0.991 — kecil tapi
# tetap diperhitungkan untuk kebenaran).
KM_PER_DEGREE_LAT = 111.32

# Label kelas seismik yang valid (nama subfolder di dalam data_dir)
VALID_LABELS = ["Multiphase", "Rockfall", "VTB", "NonEvent"]

SEED = 42
np.random.seed(SEED)

TEST_SIZE = 0.4


# ===========================================================================
# TAHAP 1 — FEATURE EXTRACTION
# ===========================================================================
#
# Catatan: ekstraksi fitur untuk TRAINING dan PREDIKSI menggunakan fungsi yang
# SAMA (extract_features_from_trace, lihat di bawah) yang dipanggil setelah
# preprocessing (preprocess_stream_per_event) yang SAMA PERSIS. Ini penting
# agar model dilatih pada domain sinyal yang identik dengan domain sinyal saat
# prediksi (instrument-corrected), bukan raw counts.

# ===========================================================================
# FK ANALYSIS & BEAMFORMING — v3 (GRID SEARCH)
# ===========================================================================
# Menggantikan pendekatan sebelumnya (cross-correlation antar stasiun +
# least-squares terhadap SATU stasiun referensi). Update dari pembimbing:
# sekarang FK dilakukan lewat GRID SEARCH langsung atas (Back-Azimuth,
# Slowness) — untuk tiap kombinasi kandidat, hitung normalized delay-and-sum
# beam power (semblance) dengan delay TEORITIS (bukan hasil cross-correlation
# yang rawan macet di skala amplitudo kecil), lalu ambil kombinasi yang
# memaksimalkan beam power. Tidak lagi bergantung pada satu stasiun referensi
# (REFERENCE_STATION) untuk FK — semua stasiun yang lolos dipakai relatif
# terhadap titik tengah (centroid) array.
#
# Back_Azimuth dan Slowness sekarang dihitung oleh fk_grid_search(), dan
# Beam_Power oleh beamforming() pada kombinasi optimum hasil grid search
# tsb. (lihat run_feature_extraction / run_feature_extraction_predict).

def _prepare_array_traces(stream, station_coords):
    """
    Prepare 3–5 station traces for array FK processing.

    Dataset-specific rule:
      - station code is expected in trace.stats.station
      - minimum 3 stations are required
      - common time interval is used
      - sampling rate must be consistent
      - each station is RMS-normalized
    """

    selected = []

    for tr in stream:
        sta = str(tr.stats.station).strip()

        if sta not in station_coords:
            continue

        if tr.stats.npts < 20:
            continue

        selected.append(tr.copy())

    # ---------------------------------------------------------------
    # Remove duplicate station codes.
    # Keep the longest trace.
    # ---------------------------------------------------------------

    best = {}

    for tr in selected:
        sta = str(tr.stats.station).strip()

        if (
            sta not in best
            or tr.stats.npts > best[sta].stats.npts
        ):
            best[sta] = tr

    selected = list(best.values())

    # Minimum 3 stations for 2-D array processing
    if len(selected) < 3:
        return None

    # ---------------------------------------------------------------
    # Common time interval
    # ---------------------------------------------------------------

    common_start = max(
        tr.stats.starttime
        for tr in selected
    )

    common_end = min(
        tr.stats.endtime
        for tr in selected
    )

    if common_end <= common_start:
        return None

    # ---------------------------------------------------------------
    # Sampling rate
    # ---------------------------------------------------------------

    fs_values = [
        float(tr.stats.sampling_rate)
        for tr in selected
    ]

    fs = min(fs_values)

    if fs <= 0:
        return None

    # ---------------------------------------------------------------
    # Prepare each station
    # ---------------------------------------------------------------

    data = []
    coords = []
    stations = []

    for tr in selected:

        try:

            tr.trim(
                common_start,
                common_end,
                pad=False
            )

            if abs(
                float(tr.stats.sampling_rate) - fs
            ) > 1e-6:

                tr.resample(fs)

            x = np.asarray(
                tr.data,
                dtype=float
            )

            if len(x) < 20:
                continue

            x = np.nan_to_num(
                x,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # Remove DC component
            x = x - np.mean(x)

            # -------------------------------------------------------
            # RMS normalization
            # -------------------------------------------------------

            rms = np.sqrt(
                np.mean(x ** 2)
            )

            if (
                not np.isfinite(rms)
                or rms <= 0
            ):
                continue

            x = x / rms

            data.append(x)

            coords.append(
                station_coords[
                    str(tr.stats.station).strip()
                ]
            )

            stations.append(
                str(tr.stats.station).strip()
            )

        except Exception:
            continue

    if len(data) < 3:
        return None

    # Same number of samples
    n = min(
        len(x)
        for x in data
    )

    data = np.asarray(
        [x[:n] for x in data],
        dtype=float
    )

    coords = np.asarray(
        coords,
        dtype=float
    )

    return (
        data,
        coords,
        fs,
        stations
    )


# ============================================================================
# ACTIVE FK WINDOW
# ============================================================================

def _extract_fk_window(
    data,
    fs,
    window_seconds=30.0
):
    """
    Select a common active window for FK.

    Window selection uses:
        median absolute amplitude across stations
        + 1-second smoothing

    The purpose is to focus FK on the active event rather than
    the complete 120-s interval dominated by noise/coda.
    """

    n = data.shape[1]

    if n <= 0:
        return data

    win = int(
        round(
            window_seconds * fs
        )
    )

    if (
        win <= 0
        or n <= win
    ):
        return data

    # ---------------------------------------------------------------
    # Median envelope across stations
    # ---------------------------------------------------------------

    envelope = np.median(
        np.abs(data),
        axis=0
    )

    # ---------------------------------------------------------------
    # Smooth with approximately 1-second moving average
    # ---------------------------------------------------------------

    smooth = max(
        1,
        int(round(1.0 * fs))
    )

    if smooth > 1:

        kernel = (
            np.ones(
                smooth
            )
            / smooth
        )

        envelope = np.convolve(
            envelope,
            kernel,
            mode="same"
        )

    # ---------------------------------------------------------------
    # Center of active window
    # ---------------------------------------------------------------

    center = int(
        np.argmax(envelope)
    )

    half = win // 2

    i0 = max(
        0,
        center - half
    )

    i1 = min(
        n,
        i0 + win
    )

    if i1 - i0 < win:

        i0 = max(
            0,
            i1 - win
        )

    return data[
        :,
        i0:i1
    ]


# ============================================================================
# FRACTIONAL DELAY / FOURIER PHASE SHIFT
# ============================================================================

def _fractional_shift(
    x,
    shift_samples
):
    """
    Fractional delay using Fourier phase shifting.

    Positive shift_samples corresponds to x(t + delay).
    """

    n = len(x)

    if (
        n < 4
        or abs(shift_samples) < 1e-9
    ):
        return x.copy()

    # Positive-frequency axis in cycles/sample
    freqs = np.fft.rfftfreq(n)

    # Fourier transform
    spec = np.fft.rfft(x)

    # ---------------------------------------------------------------
    # Phase shift
    # ---------------------------------------------------------------

    phase = np.exp(
        2j
        * np.pi
        * freqs
        * shift_samples
    )

    # Apply phase shift + inverse FFT
    y = np.fft.irfft(
        spec * phase,
        n=n
    )

    return y


# ============================================================================
# NORMALIZED DELAY-AND-SUM BEAM POWER
# ============================================================================

def _beam_score(
    data, coords_xy, fs, azimuth_deg, slowness_s_per_km,
    return_beam=False
):
    """
    Normalized delay-and-sum beam power for one (Back-Azimuth, Slowness)
    hypothesis. If return_beam=True, also returns the beamformed waveform.
    """
    if data.shape[0] < 3 or slowness_s_per_km <= 0:
        return (np.nan, None) if return_beam else np.nan

    az = np.deg2rad(float(azimuth_deg))

    # Propagation direction is opposite to Back-Azimuth.
    prop_e = np.sin(az + np.pi)
    prop_n = np.cos(az + np.pi)
    svec = slowness_s_per_km * np.array([prop_e, prop_n])

    center = np.mean(coords_xy, axis=0)
    rel = coords_xy - center

    # Theoretical time delay tau_i = r_i dot s
    delays = rel @ svec
    delay_samples = delays * fs

    aligned = np.asarray([
        _fractional_shift(data[i], delay_samples[i])
        for i in range(data.shape[0])
    ])

    # One representative array waveform.
    beam = np.mean(aligned, axis=0)

    # Normalized coherent beam power.
    raw_power = np.mean(beam ** 2)
    denom = np.mean(np.mean(aligned ** 2, axis=1)) + 1e-12
    normalized_power = float(raw_power / denom)

    if return_beam:
        return normalized_power, beam
    return normalized_power
# ============================================================================
# DIRECT FK GRID SEARCH
# ============================================================================

def fk_grid_search(
    stream,
    station_coords,
    azimuth_step=2.0,
    slowness_min=0.01,
    slowness_max=1.50,
    slowness_step=0.02,
    window_seconds=30.0
):
    """
    Direct FK grid search.

    Tidak memakai cross-correlation threshold.

    Yang dicari adalah pasangan:

        (Back-Azimuth, Slowness)

    yang memberikan normalized delay-and-sum beam power maksimum.

    Minimal 3 stasiun.
    """

    # ---------------------------------------------------------------
    # 1. Prepare array traces
    # ---------------------------------------------------------------

    prepared = _prepare_array_traces(
        stream,
        station_coords
    )

    if prepared is None:
        return (
            np.nan,
            np.nan,
            np.nan
        )

    data, coords_geo, fs, stations = (
        prepared
    )

    # ---------------------------------------------------------------
    # 2. Geographic coordinates -> local EN coordinates in km
    # ---------------------------------------------------------------

    lat = coords_geo[:, 0]
    lon = coords_geo[:, 1]

    lat0 = np.mean(lat)
    lon0 = np.mean(lon)

    x = (
        (lon - lon0)
        * KM_PER_DEGREE_LAT
        * np.cos(
            np.deg2rad(lat0)
        )
    )

    y = (
        (lat - lat0)
        * KM_PER_DEGREE_LAT
    )

    coords_xy = np.column_stack(
        (
            x,
            y
        )
    )

    # ---------------------------------------------------------------
    # 3. Check 2-D geometry
    # ---------------------------------------------------------------

    if (
        np.linalg.matrix_rank(
            coords_xy
            - np.mean(
                coords_xy,
                axis=0
            )
        )
        < 2
    ):
        return (
            np.nan,
            np.nan,
            np.nan
        )

    # ---------------------------------------------------------------
    # 4. Select active FK window
    # ---------------------------------------------------------------

    data_fk = _extract_fk_window(
        data,
        fs,
        window_seconds
    )

    if (
        data_fk.shape[1]
        < max(
            20,
            int(fs)
        )
    ):
        return (
            np.nan,
            np.nan,
            np.nan
        )

    # ---------------------------------------------------------------
    # 5. Initialize optimum
    # ---------------------------------------------------------------

    best_power = -np.inf
    best_baz = np.nan
    best_slow = np.nan

    # ---------------------------------------------------------------
    # 6. Define grid
    # ---------------------------------------------------------------

    azimuths = np.arange(
        0.0,
        360.0,
        float(azimuth_step)
    )

    slownesses = np.arange(
        float(slowness_min),
        float(slowness_max)
        + 0.5 * float(slowness_step),
        float(slowness_step)
    )

    # ---------------------------------------------------------------
    # 7. Direct exhaustive search
    # ---------------------------------------------------------------

    for slow in slownesses:

        for baz in azimuths:

            score = _beam_score(
                data_fk,
                coords_xy,
                fs,
                baz,
                slow
            )

            if (
                np.isfinite(score)
                and score > best_power
            ):

                best_power = score
                best_baz = float(
                    baz
                )
                best_slow = float(
                    slow
                )

    # ---------------------------------------------------------------
    # 8. Return optimum
    # ---------------------------------------------------------------

    if not np.isfinite(
        best_power
    ):
        return (
            np.nan,
            np.nan,
            np.nan
        )

    return (
        best_baz,
        best_slow,
        float(best_power)
    )


# ============================================================================
# FINAL BEAMFORMING AT FK OPTIMUM
# ============================================================================

def beamforming(
    stream, station_coords, baz, slowness,
    window_seconds=30.0, return_waveform=True
):
    """
    Final delay-and-sum beamforming at the optimum FK solution.

    return_waveform=True:
        returns (beam_waveform, beam_power, sampling_rate)

    return_waveform=False:
        returns beam_power only.
    """
    if not (
        np.isfinite(baz)
        and np.isfinite(slowness)
        and slowness > 0
    ):
        return (None, np.nan, np.nan) if return_waveform else np.nan

    prepared = _prepare_array_traces(stream, station_coords)
    if prepared is None:
        return (None, np.nan, np.nan) if return_waveform else np.nan

    data, coords_geo, fs, stations = prepared

    lat = coords_geo[:, 0]
    lon = coords_geo[:, 1]
    lat0 = np.mean(lat)
    lon0 = np.mean(lon)

    x = (
        (lon - lon0)
        * KM_PER_DEGREE_LAT
        * np.cos(np.deg2rad(lat0))
    )
    y = (lat - lat0) * KM_PER_DEGREE_LAT
    coords_xy = np.column_stack((x, y))

    # Exactly the same active window used by FK grid search.
    data_fk = _extract_fk_window(data, fs, window_seconds)

    if data_fk.shape[1] < max(20, int(fs)):
        return (None, np.nan, fs) if return_waveform else np.nan

    power, beam = _beam_score(
        data_fk, coords_xy, fs, baz, slowness, return_beam=True
    )

    if return_waveform:
        return beam, power, fs
    return power
def extract_features_from_trace(trace):
    """
    Ekstrak fitur statistik dan spektral dari satu trace.

    Perubahan: PSD dinormalisasi menjadi distribusi probabilitas
    sebelum hitung SpectralCentroid dan SpectralEntropy
    (konsisten antara training dan prediksi).
    """
    data = trace.data.astype(float)
    fs   = trace.stats.sampling_rate
    N    = len(data)

    if N < 10:
        return [0.0] * 11

    mean_val  = np.mean(data)
    std_val   = np.std(data)
    skew_val  = skew(data)
    kurt_val  = kurtosis(data)
    rms       = np.sqrt(np.mean(data ** 2))
    peak      = np.max(np.abs(data))
    energy    = np.sum(data ** 2)
    zero_cross = np.sum(np.diff(np.sign(data)) != 0)

    # FFT — ambil separuh positif
    freqs    = np.fft.rfftfreq(N, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(data))

    dom_freq = freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0.0

    # PSD normalisasi → distribusi probabilitas
    psd      = fft_vals ** 2
    psd_sum  = np.sum(psd) + 1e-12
    psd_norm = psd / psd_sum

    spectral_centroid = np.sum(freqs * psd_norm)
    spectral_entropy  = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    return [
        mean_val, std_val, skew_val, kurt_val,
        rms, peak, energy, zero_cross,
        dom_freq, spectral_centroid, spectral_entropy,
    ]


def preprocess_stream_for_features(
    st_raw,
    paz=None,
    prefilter=None,
    target_fs=None,
    fmin=None,
    fmax=None,
):
    """
    Preprocessing lengkap stream seismik sebelum ekstraksi fitur.
    Urutan: merge → detrend → interpolasi → taper →
            instrument correction → resample → bandpass filter.

    Digunakan KONSISTEN di training DAN prediksi agar fitur kompatibel.

    Parameter
    ---------
    st_raw     : obspy.Stream  — Stream mentah hasil read().
    paz        : AttribDict    — Pole-zero instrument response.
    prefilter  : list[float]   — Pre-filter untuk instrument correction
                                 [f1, f2, f3, f4] Hz cosine taper.
    target_fs  : float         — Target sampling rate (Hz).
    fmin       : float         — Frekuensi bawah bandpass (Hz).
    fmax       : float         — Frekuensi atas bandpass (Hz).

    Return
    ------
    obspy.Stream — Stream yang sudah diproses, siap untuk ekstraksi fitur.
    """
    if paz       is None: paz       = _DEFAULT_PAZ
    if prefilter is None: prefilter = _PREFILTER
    if target_fs is None: target_fs = _TARGET_FS
    if fmin      is None: fmin      = _FMIN
    if fmax      is None: fmax      = _FMAX

    st = st_raw.copy()

    # ── STEP 1: Merge gap ────────────────────────────────────────────
    st.merge(fill_value=0)

    st_out = Stream()
    for tr in st:
        try:
            # ── STEP 2: Detrend ──────────────────────────────────────
            tr.detrend("demean")
            tr.detrend("linear")

            # ── STEP 3: Interpolasi NaN / masked data ────────────────
            data = tr.data.astype(np.float64)
            if np.ma.is_masked(data):
                data = data.filled(np.nan)
            nan_mask = np.isnan(data)
            if nan_mask.any():
                idx   = np.arange(len(data))
                valid = idx[~nan_mask]
                if len(valid) >= 2:
                    data = np.interp(idx, valid, data[valid])
                elif len(valid) == 1:
                    data[:] = data[valid[0]]
                else:
                    continue   # semua NaN, skip trace
            tr.data = data.astype(np.float32)

            # ── STEP 4: Taper ────────────────────────────────────────
            tr.taper(max_percentage=0.05, type="cosine")

            # ── STEP 5: Instrument correction (PAZ) ─────────────────
            # Harus dilakukan SEBELUM filter agar respons instrumen
            # dihilangkan di seluruh rentang frekuensi, bukan setelah
            # sebagian frekuensi dibuang oleh filter.
            tr.stats.paz = paz
            tr.simulate(
                paz_remove=tr.stats.paz,
                remove_sensitivity=True,
                pre_filt=prefilter,
            )

            # ── STEP 6: Resample ke target_fs ────────────────────────
            if tr.stats.sampling_rate != target_fs:
                tr.resample(target_fs)

            # ── STEP 7: Bandpass filter ──────────────────────────────
            nyq      = tr.stats.sampling_rate / 2.0
            safe_fmax = min(fmax, nyq * 0.9)
            tr.filter(
                "bandpass",
                freqmin=fmin,
                freqmax=safe_fmax,
                corners=4,
                zerophase=True,
            )

            st_out += tr

        except Exception as e:
            print(f"   [WARN] Preprocess gagal {tr.stats.station}"
                  f".{tr.stats.channel}: {e}")
            continue

    return st_out


def preprocess_stream_per_event(
        st_raw,
        paz=None,
        prefilter=None,
        target_fs=None,
        fmin=None,
        fmax=None,
    ):
    """
    Preprocessing untuk data yang SUDAH dipotong per event.

    Perbedaan dari preprocess_stream_for_features():
    - TIDAK detrend agresif (cukup demean saja)
    - Taper lebih pendek (2%) karena sinyal sudah pendek
    - Instrument correction tetap dilakukan
    - Bandpass filter tetap dilakukan
    - Tidak ada interpolasi panjang (data sudah bersih)

    Urutan:
    1. Merge gap (fill_value=0)
    2. Detrend (demean saja)
    3. Taper (cosine 2%)
    4. Instrument correction (PAZ simulate)
    5. Resample ke target_fs
    6. Bandpass filter
    """
    if paz       is None: paz       = _DEFAULT_PAZ
    if prefilter is None: prefilter = _PREFILTER
    if target_fs is None: target_fs = _TARGET_FS
    if fmin      is None: fmin      = _FMIN
    if fmax      is None: fmax      = _FMAX

    st = st_raw.copy()
    st.merge(fill_value=0)

    st_out = Stream()
    for tr in st:
        try:
            # STEP 1: Detrend ringan (demean saja)
            # tr.detrend("demean")

            # STEP 2: Taper pendek (2% — data sudah pendek per event)
            tr.taper(max_percentage=0.02, type="cosine")

            # STEP 3: Instrument correction
            tr.stats.paz = paz
            tr.simulate(
                paz_remove=tr.stats.paz,
                remove_sensitivity=True,
                pre_filt=prefilter,
            )

            # Setelah tr.simulate(), tambahkan cek:
            max_amp = np.max(np.abs(tr.data))
            if max_amp > 1e-2:   # > 10 mm/s — tidak wajar, koreksi gagal
                print(f"   [WARN] {tr.stats.station}: amplitudo setelah koreksi "
                    f"mencurigakan ({max_amp:.2e} m/s) — mungkin PAZ salah")
            elif max_amp < 1e-12:
                print(f"   [WARN] {tr.stats.station}: amplitudo terlalu kecil "
                    f"({max_amp:.2e} m/s) — mungkin divide by zero")

            # STEP 4: Resample ke target_fs
            # if tr.stats.sampling_rate != target_fs:
            #     tr.resample(target_fs)

            # STEP 5: Bandpass filter
            # nyq       = tr.stats.sampling_rate / 2.0
            # safe_fmax = min(fmax, nyq * 0.9)
            # tr.filter(
            #     "bandpass",
            #     freqmin=fmin,
            #     freqmax=safe_fmax,
            #     corners=4,
            #     zerophase=True,
            # )

            st_out += tr

        except Exception as e:
            print(f"   [WARN] Preprocess gagal {tr.stats.station}"
                  f".{tr.stats.channel}: {e}")
            continue

    return st_out


def _discover_labeled_mseed_events(data_dir, valid_labels):
    """
    Temukan event .mseed secara rekursif dan tentukan label dari nama folder
    kelas. Mendukung struktur:
        data_training/<Label>/*.mseed
        data_training/<Label>/<subfolder>/*.mseed
        data_training/<Label>/<tanggal>/*.mseed

    Hanya folder kelas yang ada di valid_labels yang digunakan.

    Format filename dataset yang didukung secara eksplisit:
        YYYYMMDD_HHMMSS-HHMMSS_STATION.mseed
    sehingga seluruh stasiun dengan tanggal dan window waktu yang sama
    digabung sebagai SATU event array.
    """
    data_dir = os.path.abspath(data_dir)
    valid_set = {str(x).strip() for x in valid_labels}

    # Alias agar dataset yang memakai nama Regional_Earthquake tetap terbaca.
    aliases = {
        "Regional_Earthquake": "NonEvent",
        "Regional Earthquake": "NonEvent",
        "RegionalEarthquake": "NonEvent",
        "NEV": "NonEvent",
        "RE": "NonEvent",
    }

    groups = defaultdict(lambda: defaultdict(list))
    all_mseed = []
    label_dirs_found = set()

    for root, dirs, files in os.walk(data_dir):
        mseed_files = [
            f for f in files
            if f.lower().endswith((".mseed", ".msd", ".seed"))
        ]
        if not mseed_files:
            continue

        rel_parts = os.path.relpath(root, data_dir).split(os.sep)
        candidate_labels = []
        for part in rel_parts:
            if part in valid_set:
                candidate_labels.append(part)
            elif part in aliases and aliases[part] in valid_set:
                candidate_labels.append(aliases[part])

        if not candidate_labels:
            continue

        label = candidate_labels[0]
        label_dirs_found.add(label)

        for file in mseed_files:
            all_mseed.append(os.path.join(root, file))

            # Event ID SESUAI STRUKTUR NAMA FILE DATASET:                   
            #   YYYYMMDD_HHMMSS-HHMMSS_STATION.mseed                    
            # Contoh:                                                     
            #   20230903_045040-045240_RE5DE.mseed                       
            #   20230903_045040-045240_R265F.mseed                      
            # Keduanya adalah SATU event; kode stasiun berada pada token     
            # terakhir dan tidak boleh menjadi bagian dari event_id.         
            stem = os.path.splitext(file)[0]
            parts = stem.split("_")

            if len(parts) >= 3 and re.fullmatch(r"\d{8}", parts[0]) and "-" in parts[1]:
                event_id = "_".join(parts[:2])
            else:
                # Fallback untuk file dengan pola nama berbeda.
                # Jangan menggabungkan file secara sembarangan.
                event_id = os.path.join(
                    os.path.relpath(root, data_dir), stem
                )

            groups[label][event_id].append(
                os.path.join(root, file)
            )

    return groups, all_mseed, sorted(label_dirs_found)


def run_feature_extraction(
    data_dir, output_csv, station_coords, valid_labels=None, use_fk=True
):
    """
    FEATURE EXTRACTION BERBASIS WAVEFORM HASIL BEAMFORMING.

    Multi-station waveform
        -> preprocessing
        -> FK grid search
        -> BAZ + Slowness optimum
        -> delay-and-sum
        -> satu waveform BEAM
        -> 11 fitur temporal/spektral

    Dengan fitur spasial:
        11 + Back_Azimuth + Slowness + Beam_Power = 14 fitur.

    Tanpa fitur spasial:
        11 fitur temporal/spektral dari waveform BEAM.

    Penting:
        fungsi ini sekarang melakukan validasi jumlah file/event sebelum
        training sehingga CSV kosong tidak diteruskan ke train_test_split.
    """
    if valid_labels is None:
        valid_labels = VALID_LABELS

    print("=" * 64)
    print("TAHAP 1: FEATURE EXTRACTION — ARRAY BEAMFORMING")
    print("=" * 64)
    print(f"  Folder data : {os.path.abspath(data_dir)}")
    print(f"  Label dicari: {valid_labels}")
    print("  Sinyal dasar: 1 waveform hasil beamforming")
    print("  Fitur output: 14 fitur (11 + 3 spasial)")
    print()

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Folder data tidak ditemukan: {os.path.abspath(data_dir)}"
        )

    groups, all_mseed, found_labels = _discover_labeled_mseed_events(
        data_dir, valid_labels
    )

    print(f"  Total file waveform ditemukan : {len(all_mseed)}")
    print(f"  Folder kelas terdeteksi       : {found_labels}")
    print(
        "  Total grup event terdeteksi  : "
        f"{sum(len(v) for v in groups.values())}"
    )

    if not all_mseed:
        # Diagnostic: tampilkan struktur folder tingkat atas.
        top_dirs = []
        for name in sorted(os.listdir(data_dir)):
            p = os.path.join(data_dir, name)
            if os.path.isdir(p):
                top_dirs.append(name)

        print("\n[ERROR] Tidak ada file waveform yang masuk ke label valid.")
        print(f"  Folder tingkat atas: {top_dirs}")
        print(f"  Label yang dicari  : {list(valid_labels)}")
        print(
            "\n  Jika nama folder kelas berbeda, jalankan kembali dengan "
            "--labels sesuai nama folder kelas di data_training."
        )
        return pd.DataFrame()

    all_features, file_names, labels = [], [], []
    total_events = sum(len(v) for v in groups.values())
    processed = 0
    skipped_read = 0
    skipped_station = 0
    skipped_preprocess = 0
    skipped_fk = 0
    skipped_beam = 0

    for label in valid_labels:
        if label not in groups:
            continue

        for event_id, files in groups[label].items():
            processed += 1

            st_raw = Stream()
            for f in files:
                try:
                    st_raw += read(f)
                except Exception as e:
                    print(
                        f"  [WARN] Gagal membaca {os.path.basename(f)}: {e}"
                    )

            if len(st_raw) < 3:
                skipped_station += 1
                print(
                    f"  [SKIP {processed}/{total_events}] {label}/{event_id}: "
                    f"{len(st_raw)} trace terbaca (<3)."
                )
                continue

            try:
                st = preprocess_stream_per_event(st_raw)
            except Exception as e:
                skipped_preprocess += 1
                print(
                    f"  [SKIP {processed}/{total_events}] {label}/{event_id}: "
                    f"preprocessing error: {e}"
                )
                continue

            if len(st) < 3:
                skipped_preprocess += 1
                print(
                    f"  [SKIP {processed}/{total_events}] {label}/{event_id}: "
                    f"setelah preprocessing hanya {len(st)} trace (<3)."
                )
                continue

            # FK grid search.
            baz, slowness, fk_power = fk_grid_search(
                st, station_coords
            )

            if not (
                np.isfinite(baz)
                and np.isfinite(slowness)
                and np.isfinite(fk_power)
            ):
                skipped_fk += 1
                print(
                    f"  [SKIP {processed}/{total_events}] {label}/{event_id}: "
                    "FK gagal."
                )
                continue

            # Final beamforming pada BAZ/slowness optimum.
            beam, beam_power, beam_fs = beamforming(
                st,
                station_coords,
                baz,
                slowness,
                return_waveform=True,
            )

            if (
                beam is None
                or len(beam) < 10
                or not np.isfinite(beam_power)
                or not np.isfinite(beam_fs)
                or beam_fs <= 0
            ):
                skipped_beam += 1
                print(
                    f"  [SKIP {processed}/{total_events}] {label}/{event_id}: "
                    "waveform beam tidak valid."
                )
                continue

            # Jadikan waveform beam sebagai ObsPy Trace.
            beam_trace = st[0].copy()
            beam_trace.data = np.asarray(beam, dtype=np.float64)
            beam_trace.stats.sampling_rate = float(beam_fs)
            beam_trace.stats.station = "BEAM"
            beam_trace.stats.channel = "BEAM"

            # 11 fitur dasar diekstraksi dari SATU waveform beam.
            feat = extract_features_from_trace(beam_trace)

            if len(feat) != 11 or not np.all(np.isfinite(feat)):
                print(
                    f"  [SKIP {processed}/{total_events}] {label}/{event_id}: "
                    "fitur temporal/spektral tidak valid."
                )
                continue

            # Tiga fitur spasial.
            feat.extend([
                float(baz),
                float(slowness),
                float(beam_power),
            ])

            all_features.append(feat)
            file_names.append(event_id)
            labels.append(label)

            if processed % 10 == 0 or processed == total_events:
                print(
                    f"  Proses {processed}/{total_events} | "
                    f"berhasil={len(all_features)}"
                )

    columns = [
        "Mean", "Std", "Skewness", "Kurtosis",
        "RMS", "Peak", "Energy", "Zero_Cross",
        "Dominant_Freq", "SpectralCentroid", "SpectralEntropy",
        "Back_Azimuth", "Slowness", "Beam_Power",
    ]

    if len(all_features) == 0:
        print("\n" + "=" * 64)
        print("[ERROR] FEATURE EXTRACTION MENGHASILKAN 0 EVENT.")
        print("=" * 64)
        print(f"  Grup event          : {total_events}")
        print(f"  Skip <3 station     : {skipped_station}")
        print(f"  Skip preprocessing  : {skipped_preprocess}")
        print(f"  Skip FK             : {skipped_fk}")
        print(f"  Skip beamforming    : {skipped_beam}")
        print(
            "\nPipeline DIHENTIKAN agar tidak meneruskan CSV kosong "
            "ke train_test_split."
        )
        return pd.DataFrame()

    df = pd.DataFrame(all_features, columns=columns)
    df.insert(0, "Event", file_names)
    df.insert(1, "Label", labels)

    os.makedirs(
        os.path.dirname(os.path.abspath(output_csv)),
        exist_ok=True
    )
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 64)
    print("FEATURE EXTRACTION SELESAI")
    print("=" * 64)
    print(f"  Event berhasil : {len(df)}")
    print("  Jumlah fitur   : 14")
    print("  Distribusi kelas:")
    print(df["Label"].value_counts().to_string())
    print(f"  CSV             : {os.path.abspath(output_csv)}")
    print("=" * 64)

    return df
# ===========================================================================
# PREPROCESSING PIPELINE UNTUK DATA PREDIKSI
# ===========================================================================

# PAZ (Poles and Zeros) default instrumen RSAM/seismometer Merapi
# Sesuaikan nilai ini dengan spesifikasi instrumen yang digunakan
_DEFAULT_PAZ = AttribDict({
    'poles': [
        -1 + 3.03j,
        -1 - 3.03j,
        -3.03 + 666.67j,
        -3.03 - 666.67j
    ],
    'zeros': [0j, 0j, 0j],
    'sensitivity': 360000000,
    'gain': 693.0
})

_PREFILTER = [0.1, 0.5, 10.0, 20.0]   # [f1, f2, f3, f4] cosine taper Hz
_TARGET_FS  = 100.0                     # Hz
# Data mentah (.mseed) sudah di-bandpass 0.8-1.8 Hz SEBELUM masuk pipeline
# ini, jadi _FMIN/_FMAX di sini diselaraskan ke band yang sama supaya
# konsisten dipakai di tempat lain (mis. plot spektrum, label sumbu) —
# bandpass filter di preprocess_stream_per_event() SENGAJA tetap nonaktif
# (lihat komentar di fungsi tsb.) karena data sudah difilter di luar
# pipeline; filter ulang di sini dianggap redundan.
_FMIN       = 0.8                       # Hz — band sinyal riil di data
_FMAX       = 1.8                       # Hz — band sinyal riil di data


def run_feature_extraction_predict(
    data_dir, output_csv, station_coords, already_cut=False
):
    """
    Ekstraksi fitur prediksi/blind data dengan alur yang sama:

    waveform array -> preprocessing -> FK -> beamforming ->
    satu waveform BEAM -> 11 fitur + 3 fitur spasial.

    Output CSV selalu 14 fitur. Model 11-fitur akan mengambil hanya
    11 kolom yang tersimpan pada model.
    """
    print("=" * 60)
    print("EKSTRAKSI FITUR PREDIKSI — ARRAY BEAMFORMING")
    print("=" * 60)
    print(f"  Folder data  : {data_dir}")
    print(f"  Output CSV   : {output_csv}")
    print(
        f"  Mode         : "
        f"{'Per-Event (already cut)' if already_cut else 'Daily SEED'}"
    )
    print("  Sinyal dasar : waveform hasil beamforming array")
    print()

    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".mseed") or (
                "." in f
                and not f.lower().endswith(".csv")
                and not f.lower().endswith(".txt")
            ):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print(f"  [ERROR] Tidak ada file .mseed di: {data_dir}")
        return pd.DataFrame()

    print(f"  Total file ditemukan: {len(all_files)}")

    event_groups = defaultdict(list)

    for filepath in all_files:
        fname = os.path.basename(filepath)
        parts_underscore = fname.split("_")
        parts_dot = fname.replace(".mseed", "").split(".")

        if already_cut:
            if (
                len(parts_underscore) >= 4
                and len(parts_underscore[0]) == 8
                and parts_underscore[0].isdigit()
            ):
                event_id = "_".join(parts_underscore[:3])
            elif len(parts_dot) >= 7:
                event_id = f"{parts_dot[-2]}.{parts_dot[-1]}"
            else:
                event_id = fname.replace(".mseed", "")
        else:
            if (
                len(parts_underscore) >= 3
                and len(parts_underscore[0]) == 8
            ):
                event_id = "_".join(parts_underscore[:3])
            elif len(parts_dot) >= 7:
                event_id = f"{parts_dot[-2]}.{parts_dot[-1]}"
            elif len(parts_dot) >= 2:
                event_id = f"{parts_dot[-2]}.{parts_dot[-1]}"
            else:
                event_id = fname

        event_groups[event_id].append(filepath)

    print(f"  Total event (grup): {len(event_groups)}")
    print()

    all_features, event_names = [], []
    total = len(event_groups)
    processed = 0

    preproc_fn = (
        preprocess_stream_per_event
        if already_cut
        else preprocess_stream_for_features
    )

    for event_id, files in event_groups.items():

        st_raw = Stream()
        for f in files:
            try:
                st_raw += read(f)
            except Exception as e:
                print(f"   [WARN] Gagal baca {os.path.basename(f)}: {e}")

        if len(st_raw) < 3:
            print(f"   [SKIP] {event_id} — minimal 3 trace diperlukan.")
            processed += 1
            continue

        st = preproc_fn(st_raw)

        if len(st) < 3:
            print(f"   [SKIP] {event_id} — setelah preprocessing <3 trace.")
            processed += 1
            continue

        baz, slowness, fk_power = fk_grid_search(st, station_coords)

        if not (
            np.isfinite(baz)
            and np.isfinite(slowness)
            and np.isfinite(fk_power)
        ):
            print(f"   [SKIP] {event_id} — FK gagal.")
            processed += 1
            continue

        beam, beam_power, beam_fs = beamforming(
            st,
            station_coords,
            baz,
            slowness,
            return_waveform=True,
        )

        if (
            beam is None
            or len(beam) < 10
            or not np.isfinite(beam_power)
            or not np.isfinite(beam_fs)
            or beam_fs <= 0
        ):
            print(f"   [SKIP] {event_id} — waveform beam tidak valid.")
            processed += 1
            continue

        beam_trace = st[0].copy()
        beam_trace.data = np.asarray(beam, dtype=np.float64)
        beam_trace.stats.sampling_rate = float(beam_fs)
        beam_trace.stats.station = "BEAM"
        beam_trace.stats.channel = "BEAM"

        feat = extract_features_from_trace(beam_trace)

        if len(feat) != 11:
            print(f"   [SKIP] {event_id} — fitur dasar bukan 11.")
            processed += 1
            continue

        feat.extend([
            float(baz),
            float(slowness),
            float(beam_power),
        ])

        all_features.append(feat)
        event_names.append(event_id)

        processed += 1
        if processed % 10 == 0 or processed == total:
            print(f"  Proses {processed}/{total} event...")

    if not all_features:
        print("  [ERROR] Tidak ada fitur yang berhasil diekstrak.")
        return pd.DataFrame()

    columns = [
        "Mean", "Std", "Skewness", "Kurtosis",
        "RMS", "Peak", "Energy", "Zero_Cross",
        "Dominant_Freq", "SpectralCentroid", "SpectralEntropy",
        "Back_Azimuth", "Slowness", "Beam_Power",
    ]

    df = pd.DataFrame(all_features, columns=columns)
    df.insert(0, "Event", event_names)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print()
    print(f"  Total baris : {len(df)}")
    print("  Jumlah fitur: 14")
    print(f"✅ CSV disimpan: {output_csv}")

    return df
# ===========================================================================
# TAHAP 2 — MODEL TRAINING (SVM + XGBoost + SMOTE + GridSearchCV)
# ===========================================================================

def build_pipelines(n_classes):
    """
    Membuat dua pipeline ML yang siap dilatih: SVM-RBF dan XGBoost.

    Setiap pipeline terdiri dari langkah-langkah berurutan:
      SVM  : SMOTE → StandardScaler → SVC(kernel=rbf)
      XGB  : SMOTE → XGBClassifier

    Mengapa pipeline?
    -----------------
    Pipeline memastikan bahwa setiap langkah preprocessing (SMOTE, scaling)
    hanya diterapkan pada data training di dalam setiap fold cross-validation,
    sehingga mencegah data leakage.

    Parameter
    ---------
    n_classes : int
        Jumlah kelas unik dalam dataset (untuk konfigurasi XGBoost).

    Return
    ------
    tuple (svm_pipe, xgb_pipe, svm_grid, xgb_grid)
        Pipeline dan hyperparameter grid untuk GridSearchCV.

    Catatan SMOTE
    -------------
    SMOTE (Synthetic Minority Over-sampling Technique) menghasilkan sampel
    sintetis dari kelas minoritas dengan interpolasi di ruang fitur antar
    tetangga terdekat. Ini lebih baik dari random oversampling karena
    menambah keragaman data.

    Hyperparameter yang Di-tune
    ---------------------------
    SVM  : C (regularisasi) dan gamma (lebar kernel RBF).
           C tinggi = margin ketat (rentan overfitting).
           gamma kecil = radius kernel luas (lebih general).
    XGB  : n_estimators, max_depth, learning_rate, subsample, colsample_bytree.
    """
    svm_pipe = ImbPipeline(steps=[
        ("smote",  SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("model",  SVC(
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=42
        )),
    ])

    xgb_pipe = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    svm_grid = {
        "model__C":     [0.1, 1, 10],
        "model__gamma": ["scale", "auto"],
    }

    xgb_grid = {
        "model__n_estimators":     [200, 400],
        "model__max_depth":        [4, 6, 8],
        "model__learning_rate":    [0.05, 0.1],
        "model__subsample":        [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }

    return svm_pipe, xgb_pipe, svm_grid, xgb_grid


def run_training(output_csv, model_output, test_size=TEST_SIZE, random_state=42,
                  blind_size=0.0, blind_seed=99, data_dir=None, output_plots=None,
                  tag=None, use_fk=True):
    """
    Melatih model SVM dan XGBoost dengan GridSearchCV + StratifiedKFold.

    Alur lengkap:
      1. Load dataset CSV hasil ekstraksi fitur.
      2. Encode label string ke integer (LabelEncoder).
      3. (Opsional) Sisihkan blind holdout SEBELUM split apa pun.
      4. Stratified train/test split — mempertahankan proporsi kelas.
      5. Bangun pipeline SVM & XGBoost (dengan SMOTE).
      6. Cari hyperparameter terbaik via GridSearchCV
         (5-fold Stratified CV, scoring=f1_macro).
      7. Evaluasi model terbaik pada test set.
      8. (Opsional) Plot contoh waveform data training per kelas.
      9. Simpan model, label encoder, dan data test ke .joblib.

    Parameter
    ---------
    output_csv   : str
        Path CSV hasil feature extraction.
    model_output : str
        Folder penyimpanan model (.joblib).
    test_size    : float
        Proporsi data testing (default 0.4 = 40%), dihitung dari sisa data
        SETELAH blind set dikeluarkan (jika blind_size > 0).
    random_state : int
        Seed untuk reproducibility train/test split.
    blind_size   : float
        Proporsi seluruh dataset yang disisihkan sebagai blind holdout
        SEBELUM train/test split dilakukan. Default 0.0 — TIDAK ada blind
        holdout, seluruh data dipakai untuk train/test (dipakai ketika blind
        test akan memakai dataset terpisah lewat run_blind_test_new_data()).
        Isi dengan nilai > 0 (mis. 0.15) hanya jika ingin blind test dari
        potongan dataset yang SAMA — partisi ini disimpan di .joblib agar
        run_blind_test() bisa memuatnya kembali, menjamin blind test benar-benar
        belum pernah dilihat model.
    blind_seed   : int
        Seed untuk memisahkan blind set. Harus konsisten dengan seed yang
        dipakai run_blind_test() (default 99), tapi karena partisi disimpan
        di .joblib, run_blind_test() tidak perlu mengulang split ini lagi.
    data_dir     : str | None
        Folder data mentah .mseed (struktur subfolder label). Jika diisi
        (bersama output_plots), fungsi ini akan:
          1. Plot 1 contoh waveform+spektrum per kelas dari TRAINING set
             (plot_training_sample_waveforms() → training_sample_waveforms_{tag}.png)
          2. Plot 1 contoh waveform+spektrum per kelas dari TESTING set
             (plot_testing_sample_waveforms() → testing_sample_waveforms_{tag}.png)
          3. Simpan waveform+spektrum SETIAP event (mis. 275 event) ke
             subfolder per kelas, dipisah training/testing SESUAI split
             skenario ini (save_all_dataset_waveforms() →
             dataset_waveforms_{tag}/training|testing/{Label}/{Event}.png)
        Jika None, ketiga langkah ini dilewati.
    output_plots : str | None
        Folder simpan plot/waveform di atas. Diabaikan jika data_dir None.
    tag          : str | None
        Nama split untuk penamaan file plot (mis. "split_60_40"). Default:
        nama folder terakhir dari model_output.
    use_fk       : bool
        True (default): pakai semua kolom numerik di CSV apa adanya. False:
        kolom Back_Azimuth/Slowness/Beam_Power (jika ada di CSV) DIBUANG
        sebelum training — berguna kalau CSV diekstrak dengan use_fk=True
        tapi Anda ingin membandingkan model dengan/tanpa fitur FK tanpa
        ekstraksi ulang.

    Return
    ------
    tuple (results_dict, label_encoder, feature_cols, X_test, y_test)

    Catatan Scoring
    ---------------
    Scoring GridSearch menggunakan "f1_macro" (rata-rata F1 per kelas, tanpa
    memperhatikan support) agar evaluasi tidak bias terhadap kelas mayoritas.
    Ini penting untuk dataset seismik yang umumnya tidak seimbang.
    """
    if tag is None:
        tag = os.path.basename(os.path.normpath(model_output))

    print()
    print("=" * 60)
    print("TAHAP 2: PELATIHAN MODEL")
    print("=" * 60)
    print(f"  Input CSV    : {output_csv}")
    print(f"  Output model : {model_output}")
    print(f"  Test size    : {test_size * 100:.0f}%")
    if blind_size and blind_size > 0:
        print(f"  Blind size   : {blind_size * 100:.0f}% (disisihkan duluan, seed={blind_seed})")
    else:
        print("  Blind size   : 0% — blind holdout dilewati, seluruh data dipakai train/test.")
    print()

    if not os.path.exists(output_csv):
        raise FileNotFoundError(
            f"CSV fitur tidak ditemukan: {os.path.abspath(output_csv)}"
        )

    df = pd.read_csv(output_csv)

    # Jangan pernah meneruskan dataset kosong ke train_test_split.
    if df.empty or len(df) == 0:
        raise ValueError(
            "CSV fitur kosong (0 event). "
            "Periksa kembali FEATURE EXTRACTION dan struktur data_training."
        )

    if "Label" not in df.columns:
        raise ValueError("Kolom 'Label' tidak ditemukan pada CSV fitur.")

    if df["Label"].nunique() < 2:
        raise ValueError(
            f"Dataset hanya memiliki {df['Label'].nunique()} kelas. "
            "Minimal 2 kelas diperlukan untuk training."
        )

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    FK_COLS = ["Back_Azimuth", "Slowness", "Beam_Power"]
    if not use_fk:
        dropped = [c for c in FK_COLS if c in num_cols]
        num_cols = [c for c in num_cols if c not in FK_COLS]
        if dropped:
            print(f"  [no_fk] Kolom dibuang dari training: {dropped}")

    X        = df[num_cols].copy()
    y        = df["Label"].copy()

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("  Mapping kelas:")
    for cls, idx in zip(le.classes_, le.transform(le.classes_)):
        print(f"    {idx} → {cls}")
    print()

    # ── STEP 0: Sisihkan blind holdout SEBELUM split apa pun (OPSIONAL) ──
    # Dilakukan dengan seed BERBEDA dari random_state training agar tidak
    # tergantung pada urutan split lain, dan disimpan permanen di .joblib
    # sehingga tidak pernah ikut serta dalam train/test split maupun
    # GridSearchCV di bawah ini. Jika blind_size <= 0 (default saat ini),
    # langkah ini dilewati sepenuhnya dan SELURUH data dipakai untuk
    # train/test — dipakai ketika blind test akan memakai dataset yang
    # benar-benar terpisah (lihat run_blind_test_new_data()), bukan potongan
    # dari dataset yang sama.
    if blind_size and blind_size > 0:
        X_trainval, X_blind, y_trainval, y_blind = train_test_split(
            X, y_enc, test_size=blind_size, random_state=blind_seed, stratify=y_enc
        )
        print(f"  Blind holdout : {len(X_blind)} sampel (disisihkan, tidak dipakai training)")
    else:
        X_trainval, y_trainval = X, y_enc
        X_blind, y_blind = None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X_trainval, y_trainval, test_size=test_size, random_state=random_state,
        stratify=y_trainval
    )

    print(f"  Training set : {len(X_train)} sampel")
    print(f"  Test set     : {len(X_test)} sampel")
    print()

    if data_dir is not None and output_plots is not None:
        plot_training_sample_waveforms(
            df=df, X_train=X_train, le=le, data_dir=data_dir,
            output_dir=output_plots, tag=tag,
        )
        plot_testing_sample_waveforms(
            df=df, X_test=X_test, le=le, data_dir=data_dir,
            output_dir=output_plots, tag=tag,
        )
        save_all_dataset_waveforms(
            df=df, X_train=X_train, X_test=X_test, data_dir=data_dir,
            output_dir=output_plots, tag=tag,
        )

    svm_pipe, xgb_pipe, svm_grid, xgb_grid = build_pipelines(len(le.classes_))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    svm_search = GridSearchCV(
        svm_pipe, svm_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1
    )
    xgb_search = GridSearchCV(
        xgb_pipe, xgb_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1
    )

    print("  [1/2] Training SVM ...")
    svm_search.fit(X_train, y_train)

    print()
    print("  [2/2] Training XGBoost ...")
    xgb_search.fit(X_train, y_train)

    for name, search in [("SVM-RBF", svm_search), ("XGBoost", xgb_search)]:
        print(f"\n{'='*40}\n  {name}\n  Best params: {search.best_params_}")
        y_pred = search.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))
        print("  Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    os.makedirs(model_output, exist_ok=True)
    model_path = os.path.join(model_output, "seismic_models.joblib")
    joblib.dump({
        "label_encoder": le,
        "feature_cols":  num_cols,
        "svm_best":      svm_search.best_estimator_,
        "xgb_best":      xgb_search.best_estimator_,
        "X_train":       X_train,
        "y_train":       y_train,
        "X_test":        X_test,
        "y_test":        y_test,
        "test_size":     test_size,
        # Blind holdout — disisihkan SEBELUM train/test split & GridSearchCV,
        # disimpan agar run_blind_test() memuatnya, bukan membuat split baru
        # dari CSV penuh (yang bisa tumpang tindih dengan data training).
        "X_trainval":    X_trainval,
        "y_trainval":    y_trainval,
        "X_blind":       X_blind,
        "y_blind":       y_blind,
        "blind_size":    blind_size,
        "blind_seed":    blind_seed,
    }, model_path)

    print(f"\n✅ Model disimpan: {model_path}")
    return {
        "svm": svm_search.best_estimator_,
        "xgb": xgb_search.best_estimator_,
    }, le, num_cols, X_test, y_test



# ===========================================================================
# CLASSIFICATION RESULTS & MISCLASSIFICATION ANALYSIS
# ===========================================================================

def save_classification_and_misclassification(
    df,
    X_test,
    y_test,
    y_pred,
    y_score,
    le,
    model_name,
    output_plots,
    tag,
):
    """
    Menyimpan hasil klasifikasi setiap event TEST SET dan daftar
    misklasifikasi secara eksplisit.

    Output:
      1. classification_results_<model>_<tag>.csv
         - seluruh event test
         - true label
         - predicted label
         - correct/misclassified
         - prediction confidence
         - probabilitas setiap kelas

      2. misclassification_<model>_<tag>.csv
         - hanya event yang salah klasifikasi
         - true label
         - predicted label
         - confidence
         - probabilitas kelas

      3. misclassification_summary_<model>_<tag>.txt
         - jumlah benar/salah
         - error rate
         - pasangan true -> predicted yang paling sering muncul

    Catatan:
      X_test.index dipakai untuk menghubungkan prediksi kembali ke baris
      asli dataset sehingga Event dan Label ground-truth tidak berubah.
    """
    os.makedirs(output_plots, exist_ok=True)

    # Pastikan index test dapat dipakai untuk mengambil metadata asli.
    test_meta = df.loc[X_test.index].copy()

    y_true_int = np.asarray(y_test).astype(int)
    y_pred_int = np.asarray(y_pred).astype(int)

    true_labels = le.inverse_transform(y_true_int)
    pred_labels = le.inverse_transform(y_pred_int)

    result = pd.DataFrame(index=X_test.index)

    # Event ID dan label asli dari dataset.
    if "Event" in test_meta.columns:
        result["Event"] = test_meta["Event"].values
    else:
        result["Event"] = test_meta.index.astype(str)

    if "Label" in test_meta.columns:
        result["Original_Label"] = test_meta["Label"].values
    else:
        result["Original_Label"] = true_labels

    result["True_Label"] = true_labels
    result["Predicted_Label"] = pred_labels
    result["Correct"] = result["True_Label"].values == result["Predicted_Label"].values
    result["Classification_Status"] = np.where(
        result["Correct"],
        "Correct",
        "Misclassified"
    )

    # Confidence = probabilitas kelas yang dipilih model.
    y_score = np.asarray(y_score)

    if y_score.ndim == 2 and y_score.shape[0] == len(result):
        result["Prediction_Confidence"] = np.max(y_score, axis=1)

        # Simpan probabilitas setiap kelas.
        for j, class_name in enumerate(le.classes_):
            if j < y_score.shape[1]:
                safe_name = re.sub(
                    r"[^A-Za-z0-9_]+", "_", str(class_name)
                ).strip("_")
                result[f"Prob_{safe_name}"] = y_score[:, j]

    # Simpan beberapa fitur utama untuk membantu interpretasi error.
    # Tidak mengubah model; hanya menyalin nilai test set ke laporan.
    for col in X_test.columns:
        if col not in result.columns:
            result[f"Feature_{col}"] = X_test[col].values

    result = result.reset_index(drop=True)

    # Urutkan event yang salah di bagian atas berdasarkan confidence.
    result["Prediction_Confidence"] = pd.to_numeric(
        result["Prediction_Confidence"], errors="coerce"
    )

    all_results_path = os.path.join(
        output_plots,
        f"classification_results_{model_name}_{tag}.csv"
    )

    mis = result.loc[
        ~result["Correct"]
    ].copy()

    mis = mis.sort_values(
        "Prediction_Confidence",
        ascending=False,
        na_position="last"
    )

    mis_path = os.path.join(
        output_plots,
        f"misclassification_{model_name}_{tag}.csv"
    )

    result.to_csv(all_results_path, index=False)
    mis.to_csv(mis_path, index=False)

    # Ringkasan pasangan salah klasifikasi.
    pair_counts = (
        mis.groupby(["True_Label", "Predicted_Label"])
        .size()
        .sort_values(ascending=False)
    )

    total = len(result)
    n_correct = int(result["Correct"].sum())
    n_mis = int((~result["Correct"]).sum())
    accuracy = n_correct / total if total else np.nan
    error_rate = n_mis / total if total else np.nan

    summary_path = os.path.join(
        output_plots,
        f"misclassification_summary_{model_name}_{tag}.txt"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("SEISMIC EVENT CLASSIFICATION — MISCLASSIFICATION ANALYSIS\n")
        f.write("=" * 72 + "\n")
        f.write(f"Model              : {model_name}\n")
        f.write(f"Split tag          : {tag}\n")
        f.write(f"Test samples       : {total}\n")
        f.write(f"Correct             : {n_correct}\n")
        f.write(f"Misclassified      : {n_mis}\n")
        f.write(f"Accuracy            : {accuracy:.4f}\n")
        f.write(f"Misclassification rate: {error_rate:.4f}\n\n")

        f.write("Misclassification by true -> predicted class\n")
        f.write("-" * 72 + "\n")

        if len(pair_counts) == 0:
            f.write("No misclassified events.\n")
        else:
            for (true_cls, pred_cls), count in pair_counts.items():
                f.write(
                    f"{str(true_cls):<25} -> "
                    f"{str(pred_cls):<25} : {int(count)} event(s)\n"
                )

        f.write("\nPer-class test results\n")
        f.write("-" * 72 + "\n")

        for class_name in le.classes_:
            subset = result[result["True_Label"] == class_name]
            if len(subset) == 0:
                continue

            correct_cls = int(subset["Correct"].sum())
            wrong_cls = len(subset) - correct_cls
            recall_cls = correct_cls / len(subset)

            f.write(
                f"{str(class_name):<25} "
                f"N={len(subset):>3}  "
                f"Correct={correct_cls:>3}  "
                f"Misclassified={wrong_cls:>3}  "
                f"Recall={recall_cls:.4f}\n"
            )

    print(f"   Classification CSV : {all_results_path}")
    print(f"   Misclassification  : {mis_path}")
    print(f"   Misclassification report: {summary_path}")
    print(
        f"   {model_name}: {n_mis}/{total} test event(s) "
        f"misclassified ({error_rate*100:.2f}%)"
    )

    return result, mis


# ===========================================================================
# TAHAP 3 — EVALUASI & VISUALISASI
# ===========================================================================

def plot_confusion_matrix(y_test, y_pred, class_names, title, save_path=None):
    """
    Membuat dan menyimpan plot Confusion Matrix dalam dua tampilan side-by-side:
      - Kiri  : Nilai absolut (jumlah sampel) — ConfusionMatrixDisplay standar.
      - Kanan : Persentase per baris (row-normalized) — seberapa banyak tiap
                kelas aktual diprediksi benar/salah dalam persen.

    Normalisasi per-baris dipilih karena langsung menjawab pertanyaan:
    "Dari semua event X, berapa persen yang diklasifikasikan dengan benar?"
    Ini ekuivalen dengan Recall per kelas dan tidak terpengaruh imbalance.

    Parameter
    ---------
    y_test      : array-like  — Label aktual (integer encoded).
    y_pred      : array-like  — Label hasil prediksi model.
    class_names : list[str]   — Nama kelas sesuai urutan integer.
    title       : str         — Judul utama plot.
    save_path   : str | None  — Path simpan gambar (PNG). None = hanya tampil.
    """
    cm      = confusion_matrix(y_test, y_pred)
    cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100  # row-normalized

    # Ukuran figur menyesuaikan jumlah kelas agar label tidak bertabrakan
    n_cls   = len(class_names)
    cell_w  = max(1.6, 7.0 / n_cls)   # minimal 1.6 inch per kolom
    fig_w   = cell_w * n_cls * 2 + 2   # 2 panel + margin
    fig_h   = max(5.0, cell_w * n_cls + 1.5)

    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    # ── Panel kiri: nilai absolut ──────────────────────────────────────
    disp_abs = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp_abs.plot(
        cmap="Blues",
        xticks_rotation=40,
        ax=ax_abs,
        colorbar=False,
    )
    ax_abs.set_title("Jumlah Absolut", fontsize=11, fontweight="bold", pad=8)
    ax_abs.set_xlabel("Prediksi", fontsize=10)
    ax_abs.set_ylabel("Aktual", fontsize=10)
    ax_abs.tick_params(axis="both", labelsize=9)

    # ── Panel kanan: persentase per baris ─────────────────────────────
    im = ax_pct.imshow(cm_pct, interpolation="nearest", cmap="RdYlGn",
                       vmin=0, vmax=100)
    plt.colorbar(im, ax=ax_pct, shrink=0.82, label="% (per baris aktual)")

    # Anotasi nilai persen di setiap sel
    thresh = 50.0   # warna teks: putih untuk sel gelap, hitam untuk sel terang
    for i in range(n_cls):
        for j in range(n_cls):
            val    = cm_pct[i, j]
            color  = "white" if val > thresh else "black"
            weight = "bold"  if i == j     else "normal"    # diagonal = benar
            ax_pct.text(
                j, i, f"{val:.1f}%",
                ha="center", va="center",
                fontsize=max(8, 11 - n_cls),   # font mengecil jika kelas banyak
                color=color, fontweight=weight,
            )

    ax_pct.set_xticks(range(n_cls))
    ax_pct.set_yticks(range(n_cls))
    ax_pct.set_xticklabels(class_names, rotation=40, ha="right", fontsize=9)
    ax_pct.set_yticklabels(class_names, fontsize=9)
    ax_pct.set_xlabel("Prediksi", fontsize=10)
    ax_pct.set_ylabel("Aktual", fontsize=10)
    ax_pct.set_title("Persentase per Baris (Recall Visual)", fontsize=11,
                     fontweight="bold", pad=8)

    # Garis grid tipis antar sel
    ax_pct.set_xticks(np.arange(n_cls) - 0.5, minor=True)
    ax_pct.set_yticks(np.arange(n_cls) - 0.5, minor=True)
    ax_pct.grid(which="minor", color="white", linewidth=1.5)
    ax_pct.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Plot: {save_path}")
    plt.close()


def plot_roc_curve(y_test, y_score, class_names, title, save_path=None):
    """
    Membuat dan menyimpan plot ROC Curve (One-vs-Rest untuk multiclass).

    ROC (Receiver Operating Characteristic) Curve menggambarkan trade-off
    antara True Positive Rate (TPR / sensitivitas) dan False Positive Rate (FPR)
    pada berbagai threshold keputusan.

    AUC (Area Under the ROC Curve) mengkuantifikasi kinerja:
      - AUC = 1.0 → Klasifikasi sempurna
      - AUC = 0.5 → Tidak lebih baik dari tebak acak
      - AUC > 0.9 → Excellent untuk kebanyakan aplikasi seismik

    Untuk multiclass, digunakan strategi One-vs-Rest (OvR): setiap kelas
    dievaluasi sebagai "kelas positif" melawan semua kelas lainnya.

    Parameter
    ---------
    y_test      : array-like         — Label aktual (integer).
    y_score     : array (n, n_class) — Probabilitas prediksi per kelas.
    class_names : list[str]          — Nama kelas.
    title       : str                — Judul plot.
    save_path   : str | None         — Path simpan gambar.
    """
    y_test_bin         = label_binarize(y_test, classes=range(len(class_names)))
    fpr, tpr, roc_auc  = {}, {}, {}

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i]        = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10.colors
    for i, cls in enumerate(class_names):
        plt.plot(
            fpr[i], tpr[i], lw=2, color=colors[i % len(colors)],
            label=f"{cls} (AUC = {roc_auc[i]:.3f})"
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Plot: {save_path}")
    # plt.show()
    plt.close()


def plot_learning_curve(model, X, y_enc, title, cv, save_path=None):
    """
    Membuat dan menyimpan plot Learning Curve.

    Learning Curve menunjukkan bagaimana performa model (training & validation
    accuracy) berubah seiring bertambahnya ukuran data training. Berguna untuk
    mendiagnosis masalah bias-variance:

      Interpretasi:
      - Underfitting  : Training & validation score keduanya rendah dan hampir sama.
                        Solusi: tambah kompleksitas model atau fitur baru.
      - Overfitting   : Training score tinggi, validation score rendah (gap besar).
                        Solusi: tambah data, regularisasi, atau kurangi kompleksitas.
      - Ideal         : Keduanya tinggi dan konvergen (gap kecil).

    Parameter
    ---------
    model     : sklearn estimator  — Model yang akan dievaluasi.
    X         : array-like         — Seluruh data fitur.
    y_enc     : array-like         — Seluruh label integer encoded.
    title     : str                — Judul plot.
    cv        : cross-validator    — Objek StratifiedKFold.
    save_path : str | None         — Path simpan gambar.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y_enc, cv=cv, scoring="accuracy",
        train_sizes=np.linspace(0.5, 1.0, 5), n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = np.mean(val_scores,  axis=1)
    val_std    = np.std(val_scores,   axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", color="royalblue",  label="Training Score")
    plt.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std,
        alpha=0.15, color="royalblue"
    )
    plt.plot(train_sizes, val_mean, "o-", color="darkorange", label="Validation Score")
    plt.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std,
        alpha=0.15, color="darkorange"
    )
    plt.xlabel("Training Size", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Plot: {save_path}")
    # plt.show()
    plt.close()

def plot_learning_curve_2(model, X, y, model_name, class_names, output_dir):
    """
    Plot learning curve: skor training vs validation sebagai fungsi jumlah sampel.
    Menunjukkan apakah model underfitting, overfitting, atau sudah optimal.

    Referensi:
      - Scikit-learn documentation: Learning Curves
      - Raschka (2018), Model Evaluation, MLxtend
    """
    print(f"\n  Menghitung learning curve untuk {model_name}...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scorer = make_scorer(f1_score, average="macro", zero_division=0)

    # Buat range training sizes
    n_samples = len(X)
    min_samples = max(5 * 2, 10)
    train_sizes = np.linspace(0.1, 1.0, 10)
    # Filter ukuran yang terlalu kecil
    train_sizes = train_sizes[train_sizes * n_samples * (1 - 1/5) >= min_samples]

    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_scaled, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            random_state=SEED,
            shuffle=True,
        )
    except Exception as e:
        print(f"  [WARN] Learning curve gagal: {e}")
        return

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Learning Curve (F1-macro) ---
    ax1.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color="#2196F3")
    ax1.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color="#FF9800")
    ax1.plot(train_sizes_abs, train_mean, "o-", color="#2196F3",
             linewidth=2, markersize=5, label="Training Score")
    ax1.plot(train_sizes_abs, val_mean, "o-", color="#FF9800",
             linewidth=2, markersize=5, label="Validation Score")

    ax1.set_xlabel("Jumlah Sampel Training", fontsize=11)
    ax1.set_ylabel("F1-macro Score", fontsize=11)
    ax1.set_title(f"{model_name} — Learning Curve", fontsize=13)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Anotasi gap (indikator overfitting)
    gap = train_mean[-1] - val_mean[-1]
    ax1.annotate(
        f"Gap: {gap:.3f}",
        xy=(train_sizes_abs[-1], (train_mean[-1] + val_mean[-1]) / 2),
        fontsize=10, ha="right", color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    # --- Plot 2: Scalability (waktu tidak tersedia, gunakan gap per size) ---
    gap_per_size = train_mean - val_mean
    ax2.plot(train_sizes_abs, gap_per_size, "s-", color="#E91E63",
             linewidth=2, markersize=5)
    ax2.axhline(y=0.05, color="green", linestyle="--", alpha=0.7, label="Threshold 0.05")
    ax2.fill_between(train_sizes_abs, 0, gap_per_size, alpha=0.1, color="#E91E63")
    ax2.set_xlabel("Jumlah Sampel Training", fontsize=11)
    ax2.set_ylabel("Train-Validation Gap (F1-macro)", fontsize=11)
    ax2.set_title(f"{model_name} — Generalization Gap", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_learning_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Diagnosis otomatis
    if gap > 0.15:
        print(f"  [DIAGNOSIS] {model_name}: Kemungkinan OVERFITTING (gap={gap:.3f}).")
        print(f"              Saran: tambah data, kurangi kompleksitas model, atau gunakan regularisasi.")
    elif val_mean[-1] < 0.6:
        print(f"  [DIAGNOSIS] {model_name}: Kemungkinan UNDERFITTING (val_score={val_mean[-1]:.3f}).")
        print(f"              Saran: tambah fitur, tingkatkan kompleksitas model.")
    else:
        print(f"  [DIAGNOSIS] {model_name}: Model terlihat baik (val={val_mean[-1]:.3f}, gap={gap:.3f}).")


def plot_class_distribution(y_series, class_names, save_path=None):
    """
    Membuat plot distribusi kelas (jumlah sampel per kelas) dalam dataset.

    Visualisasi ini penting untuk:
      1. Mendeteksi class imbalance (ketidakseimbangan kelas), yang mempengaruhi
         pemilihan metode oversampling (SMOTE) dan metrik evaluasi (f1_macro).
      2. Memahami komposisi dataset seismik (seberapa banyak setiap jenis event).

    Dalam dataset Merapi: VTB biasanya dominan, diikuti Rockfall, NonEvent,
    dan Multiphase paling sedikit.

    Parameter
    ---------
    y_series    : pd.Series   — Kolom label string dari DataFrame.
    class_names : list[str]   — Urutan kelas untuk sumbu x.
    save_path   : str | None  — Path simpan gambar.
    """
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x=y_series, order=class_names, palette="Set2")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", fontsize=11
        )
    plt.title("Distribusi Kelas dalam Dataset", fontsize=13, fontweight="bold")
    plt.xlabel("Kelas", fontsize=12)
    plt.ylabel("Jumlah Sampel", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Plot: {save_path}")
    # plt.show()
    plt.close()


def _plot_sample_waveforms_for_subset(
    df, X_subset, le, data_dir, output_dir, tag,
    subset_label, file_prefix,
    station=None, random_state=42,
    paz=None, prefilter=None, target_fs=None, fmin=None, fmax=None,
):
    """
    Helper bersama untuk plot_training_sample_waveforms() dan
    plot_testing_sample_waveforms() — plot contoh waveform (+ spektrum)
    per kelas dari event yang BENAR-BENAR ada di subset (train/test) split
    ini, bukan sekadar file pertama yang ada di folder.

    Cara kerja
    ----------
    1. df.loc[X_subset.index] memetakan baris subset kembali ke Event & Label
       aslinya (index asli tetap terjaga pandas setelah train_test_split).
    2. Untuk tiap kelas, pilih 1 event secara acak dari subset ini, cari file
       .mseed stasiun referensi (REFERENCE_STATION) untuk event tersebut.
    3. Terapkan preprocess_stream_per_event() — SAMA PERSIS dengan pipeline
       training/prediksi — supaya waveform yang tampil mencerminkan sinyal
       yang benar-benar "dilihat" model.

    Parameter
    ---------
    df           : pd.DataFrame — dataset fitur lengkap (kolom Event & Label).
    X_subset     : pd.DataFrame — subset fitur (X_train atau X_test) hasil
                   train_test_split (index-nya harus tetap bagian dari df.index).
    le           : LabelEncoder — dipakai untuk urutan kelas.
    data_dir     : str  — folder data mentah .mseed (struktur subfolder label).
    output_dir   : str  — folder simpan plot.
    tag          : str  — nama split, dipakai untuk penamaan file output.
    subset_label : str  — label ditampilkan di judul plot (mis. "TRAINING").
    file_prefix  : str  — awalan nama file output (mis. "training_sample").
    station      : str | None — kode stasiun yang diplot. Default: REFERENCE_STATION.
    """
    if station is None:
        station = REFERENCE_STATION
    if not data_dir or not os.path.isdir(data_dir):
        print(f"   [WARN] data_dir tidak valid ({data_dir}) — plot sampel {subset_label.lower()} dilewati.")
        return

    if paz       is None: paz       = _DEFAULT_PAZ
    if prefilter is None: prefilter = _PREFILTER
    if target_fs is None: target_fs = _TARGET_FS
    if fmin      is None: fmin      = _FMIN
    if fmax      is None: fmax      = _FMAX

    print(f"\n  Membuat plot contoh waveform data {subset_label} per kelas [{tag}]...")

    subset_info = df.loc[X_subset.index, ["Event", "Label"]]
    class_names = list(le.classes_)
    rng         = np.random.RandomState(random_state)

    class_colors = {
        "NonEvent":   "#2196F3",
        "Multiphase": "#E91E63",
        "Rockfall":   "#FF9800",
        "VTB":        "#4CAF50",
    }

    samples = {}
    for cls in class_names:
        candidates = subset_info.loc[subset_info["Label"] == cls, "Event"].unique().tolist()
        if not candidates:
            print(f"   [WARN] Tidak ada sampel kelas {cls} di {subset_label.lower()} set split ini.")
            continue
        rng.shuffle(candidates)
        found = None
        for ev in candidates:
            direct = os.path.join(data_dir, cls, f"{ev}_{station}.mseed")
            if os.path.exists(direct):
                found = (ev, direct)
                break
            matches = glob.glob(os.path.join(data_dir, cls, f"{ev}*{station}*.mseed"))
            if matches:
                found = (ev, matches[0])
                break
        if found:
            samples[cls] = found
        else:
            print(f"   [WARN] File stasiun {station} tidak ditemukan untuk kelas {cls} "
                  f"({len(candidates)} event {subset_label.lower()} dicoba).")

    if not samples:
        print(f"   [WARN] Tidak ada sampel {subset_label.lower()} yang bisa diplot.")
        return

    n_classes = len(samples)
    fig = plt.figure(figsize=(15, 3.8 * n_classes))
    gs  = gridspec.GridSpec(n_classes, 2, width_ratios=[3, 1],
                            hspace=0.6, wspace=0.3)

    for idx, (cls, (event_id, filepath)) in enumerate(samples.items()):
        color = class_colors.get(cls, "#607D8B")

        try:
            st_raw = read(filepath)
            st = preprocess_stream_per_event(
                st_raw, paz=paz, prefilter=prefilter,
                target_fs=target_fs, fmin=fmin, fmax=fmax,
            )
            if len(st) == 0:
                raise ValueError("preprocessing menghasilkan stream kosong")
            tr   = st[0]
            data = tr.data.astype(np.float64)
            sr   = tr.stats.sampling_rate
        except Exception as e:
            print(f"   [WARN] Gagal memuat sampel {subset_label.lower()} {cls}: {e}")
            continue

        t = np.arange(len(data)) / sr

        # --- Subplot kiri: waveform ---
        ax_wave = fig.add_subplot(gs[idx, 0])
        ax_wave.plot(t, data, color=color, linewidth=0.7, alpha=0.9)
        ax_wave.set_title(
            f"{cls}  —  Event: {event_id}  (stasiun {station}, {subset_label.lower()} set)",
            # pad=20 (bukan pad kecil) supaya judul loc="left" tidak
            # bertumpuk dengan notasi skala sumbu-Y (mis. "1e-12") yang
            # dirender ObsPy/matplotlib di pojok kiri-atas — amplitudo
            # trace ber-orde sangat kecil (m/s hasil instrument correction)
            # sehingga notasi ini hampir selalu muncul.
            fontsize=10, fontweight="bold", loc="left", pad=20
        )
        ax_wave.set_xlabel("Waktu (detik)", fontsize=8)
        ax_wave.set_ylabel("Amplitudo (m/s)", fontsize=8)
        ax_wave.tick_params(labelsize=7)
        ax_wave.grid(True, alpha=0.3)
        ax_wave.margins(x=0)
        ax_wave.annotate(
            f"SR = {sr:.0f} Hz  |  Durasi = {len(data)/sr:.1f} s",
            xy=(0.99, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=7.5, color="gray",
        )

        # --- Subplot kanan: spektrum frekuensi ---
        ax_spec = fig.add_subplot(gs[idx, 1])
        n_pts   = len(data)
        fft_mag = np.abs(np.fft.rfft(data))
        freqs   = np.fft.rfftfreq(n_pts, d=1.0 / sr)

        mask = freqs <= (fmax + 5)
        ax_spec.plot(freqs[mask], fft_mag[mask],
                     color=color, linewidth=0.8, alpha=0.9)

        ax_spec.set_xlabel("Frekuensi (Hz)", fontsize=8)
        ax_spec.set_ylabel("|FFT|", fontsize=8)
        ax_spec.set_title("Spektrum", fontsize=9, pad=4)
        ax_spec.tick_params(labelsize=7)
        ax_spec.grid(True, alpha=0.3)
        ax_spec.margins(x=0)

    fig.suptitle(
        f"Contoh Waveform Data {subset_label} per Kelas — {tag} (stasiun {station})",
        fontsize=13, fontweight="bold", y=1.01
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{file_prefix}_waveforms_{tag}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot: {path}")


def plot_training_sample_waveforms(
    df, X_train, le, data_dir, output_dir, tag,
    station=None, random_state=42,
    paz=None, prefilter=None, target_fs=None, fmin=None, fmax=None,
):
    """
    Plot contoh waveform (+ spektrum) per kelas dari event yang BENAR-BENAR
    dipakai di TRAINING set split ini. Lihat _plot_sample_waveforms_for_subset()
    untuk detail cara kerja. Output: training_sample_waveforms_{tag}.png
    """
    _plot_sample_waveforms_for_subset(
        df, X_train, le, data_dir, output_dir, tag,
        subset_label="TRAINING", file_prefix="training_sample",
        station=station, random_state=random_state,
        paz=paz, prefilter=prefilter, target_fs=target_fs, fmin=fmin, fmax=fmax,
    )


def plot_testing_sample_waveforms(
    df, X_test, le, data_dir, output_dir, tag,
    station=None, random_state=42,
    paz=None, prefilter=None, target_fs=None, fmin=None, fmax=None,
):
    """
    Plot contoh waveform (+ spektrum) per kelas dari event yang BENAR-BENAR
    dipakai di TESTING set split ini. Lihat _plot_sample_waveforms_for_subset()
    untuk detail cara kerja. Output: testing_sample_waveforms_{tag}.png
    """
    _plot_sample_waveforms_for_subset(
        df, X_test, le, data_dir, output_dir, tag,
        subset_label="TESTING", file_prefix="testing_sample",
        station=station, random_state=random_state,
        paz=paz, prefilter=prefilter, target_fs=target_fs, fmin=fmin, fmax=fmax,
    )


def _save_waveforms_for_event_list(
    events, cls_col, data_dir, out_root, station,
    paz, prefilter, target_fs, fmin, fmax, subset_name,
):
    """
    Helper internal save_all_dataset_waveforms(): simpan waveform+spektrum
    untuk daftar event (DataFrame kolom Event & Label) ke out_root/{Label}/{Event}.png.
    """
    class_colors = {
        "NonEvent":   "#2196F3",
        "Multiphase": "#E91E63",
        "Rockfall":   "#FF9800",
        "VTB":        "#4CAF50",
    }

    total = len(events)
    print(f"\n  Menyimpan waveform+spektrum {subset_name.upper()} ({total} event)...")
    os.makedirs(out_root, exist_ok=True)

    saved, failed = 0, 0
    for _, row in events.iterrows():
        event_id, cls = row["Event"], row[cls_col]
        color = class_colors.get(cls, "#607D8B")

        direct = os.path.join(data_dir, cls, f"{event_id}_{station}.mseed")
        if os.path.exists(direct):
            filepath = direct
        else:
            matches = glob.glob(os.path.join(data_dir, cls, f"{event_id}*{station}*.mseed"))
            filepath = matches[0] if matches else None

        if filepath is None:
            print(f"   [WARN] File stasiun {station} tidak ditemukan untuk event {event_id} ({cls}).")
            failed += 1
            continue

        try:
            st_raw = read(filepath)
            st = preprocess_stream_per_event(
                st_raw, paz=paz, prefilter=prefilter,
                target_fs=target_fs, fmin=fmin, fmax=fmax,
            )
            if len(st) == 0:
                raise ValueError("preprocessing menghasilkan stream kosong")
            tr   = st[0]
            data = tr.data.astype(np.float64)
            sr   = tr.stats.sampling_rate
        except Exception as e:
            print(f"   [WARN] Gagal memuat waveform {event_id} ({cls}): {e}")
            failed += 1
            continue

        t = np.arange(len(data)) / sr
        fig, (ax_wave, ax_spec) = plt.subplots(1, 2, figsize=(13, 3.2),
                                               gridspec_kw={"width_ratios": [3, 1]})

        ax_wave.plot(t, data, color=color, linewidth=0.7, alpha=0.9)
        # pad=20 — lihat catatan di _plot_sample_waveforms_for_subset()
        # soal tumpang-tindih judul loc="left" dengan notasi skala sumbu-Y.
        ax_wave.set_title(f"{cls} — Event: {event_id} (stasiun {station}, {subset_name})",
                          fontsize=10, fontweight="bold", loc="left", pad=20)
        ax_wave.set_xlabel("Waktu (detik)", fontsize=8)
        ax_wave.set_ylabel("Amplitudo (m/s)", fontsize=8)
        ax_wave.tick_params(labelsize=7)
        ax_wave.grid(True, alpha=0.3)
        ax_wave.margins(x=0)

        fft_mag = np.abs(np.fft.rfft(data))
        freqs   = np.fft.rfftfreq(len(data), d=1.0 / sr)
        mask    = freqs <= (fmax + 5)
        ax_spec.plot(freqs[mask], fft_mag[mask], color=color, linewidth=0.8, alpha=0.9)
        ax_spec.set_xlabel("Frekuensi (Hz)", fontsize=8)
        ax_spec.set_ylabel("|FFT|", fontsize=8)
        ax_spec.set_title("Spektrum", fontsize=9)
        ax_spec.tick_params(labelsize=7)
        ax_spec.grid(True, alpha=0.3)
        ax_spec.margins(x=0)

        plt.tight_layout()

        cls_dir = os.path.join(out_root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        safe_name = "".join(c for c in str(event_id) if c.isalnum() or c in "_-.")
        fig_path = os.path.join(cls_dir, f"{safe_name}.png")
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        saved += 1

        if saved % 25 == 0 or (saved + failed) == total:
            print(f"   Proses {saved + failed}/{total} event ({subset_name})...")

    print(f"   ✅ {subset_name}: {saved} berhasil, {failed} gagal → {out_root}")


def save_all_dataset_waveforms(
    df, X_train, X_test, data_dir, output_dir, tag,
    station=None, paz=None, prefilter=None, target_fs=None, fmin=None, fmax=None,
):
    """
    Menyimpan waveform + spektrum INDIVIDUAL untuk SETIAP event yang dipakai
    di split ini — bukan cuma 1 contoh per kelas seperti plot_training/testing_
    sample_waveforms(), tapi SELURUH event (mis. total 275: 71 Multiphase +
    49 NonEvent + 84 RockFall + 71 VTB) — dipisah ke subfolder training/
    dan testing/ SESUAI keanggotaan event tsb di split (X_train/X_test) untuk
    skenario ini, bukan digabung jadi satu folder.

    Struktur output
    ---------------
    output_dir/dataset_waveforms_{tag}/
    ├── training/{Label}/{Event}.png   ← event yang dipakai untuk training
    └── testing/{Label}/{Event}.png    ← event yang dipakai untuk testing

    Preprocessing memakai preprocess_stream_per_event() — SAMA PERSIS
    dengan pipeline ekstraksi fitur — supaya waveform yang tersimpan
    mencerminkan sinyal yang benar-benar dipakai model.

    Parameter
    ---------
    df         : pd.DataFrame — dataset fitur lengkap (kolom Event & Label),
                 1 baris = 1 event (lihat run_feature_extraction()).
    X_train    : pd.DataFrame — subset fitur training hasil train_test_split
                 (index-nya harus tetap bagian dari df.index).
    X_test     : pd.DataFrame — subset fitur testing hasil train_test_split.
    data_dir   : str  — folder data mentah .mseed (struktur subfolder label).
    output_dir : str  — folder simpan plot.
    tag        : str  — nama split, dipakai untuk penamaan sub-folder output.
    station    : str | None — kode stasiun yang diplot. Default: REFERENCE_STATION.
    """
    if station is None:
        station = REFERENCE_STATION
    if not data_dir or not os.path.isdir(data_dir):
        print(f"   [WARN] data_dir tidak valid ({data_dir}) — simpan waveform dataset dilewati.")
        return

    if paz       is None: paz       = _DEFAULT_PAZ
    if prefilter is None: prefilter = _PREFILTER
    if target_fs is None: target_fs = _TARGET_FS
    if fmin      is None: fmin      = _FMIN
    if fmax      is None: fmax      = _FMAX

    base_dir = os.path.join(output_dir, f"dataset_waveforms_{tag}")

    train_events = df.loc[X_train.index, ["Event", "Label"]].drop_duplicates().reset_index(drop=True)
    test_events  = df.loc[X_test.index,  ["Event", "Label"]].drop_duplicates().reset_index(drop=True)

    _save_waveforms_for_event_list(
        train_events, "Label", data_dir, os.path.join(base_dir, "training"),
        station, paz, prefilter, target_fs, fmin, fmax, subset_name="training",
    )
    _save_waveforms_for_event_list(
        test_events, "Label", data_dir, os.path.join(base_dir, "testing"),
        station, paz, prefilter, target_fs, fmin, fmax, subset_name="testing",
    )


def plot_data_distribution_2(df, feature_cols, class_col, output_dir, level_name=""):
    """
    Visualisasi distribusi data training secara komprehensif.
    Menghasilkan 4 subplot:
      1. Distribusi jumlah sampel per kelas (bar + pie)
      2. Distribusi fitur-fitur penting (violin plot)
      3. Korelasi antar fitur top (heatmap)
      4. PCA 2D scatter plot per kelas

    Referensi:
      - Habbak et al. (2024), Nature Sci. Rep. — pentingnya analisis distribusi fitur
      - Sidik et al. (2023), Acta Geophysica — visualisasi data seismik vulkanik
    """
    print(f"\n  Membuat visualisasi distribusi data {level_name}...")

    class_names = sorted(df[class_col].unique())
    class_counts = df[class_col].value_counts().reindex(class_names)
    n_classes = len(class_names)
    colors = plt.cm.Set2(np.linspace(0, 0.85, n_classes))

    # ===== FIGURE 1: Distribusi kelas + ringkasan =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1a. Bar chart distribusi kelas
    bars = axes[0].bar(range(n_classes), class_counts.values, color=colors, edgecolor="gray", linewidth=0.5)
    axes[0].set_xticks(range(n_classes))
    axes[0].set_xticklabels(class_names, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Jumlah Sampel")
    axes[0].set_title(f"Distribusi Kelas {level_name}")
    for i, (bar, cnt) in enumerate(zip(bars, class_counts.values)):
        pct = 100 * cnt / len(df)
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    # 1b. Pie chart
    wedges, texts, autotexts = axes[1].pie(
        class_counts.values, labels=class_names, autopct="%1.1f%%",
        colors=colors, startangle=90, textprops={"fontsize": 9},
    )
    axes[1].set_title(f"Proporsi Kelas {level_name}")

    # 1c. Imbalance ratio
    max_count = class_counts.max()
    ratios = max_count / class_counts.values
    axes[2].barh(range(n_classes), ratios, color=colors, edgecolor="gray", linewidth=0.5)
    axes[2].set_yticks(range(n_classes))
    axes[2].set_yticklabels(class_names, fontsize=9)
    axes[2].set_xlabel("Imbalance Ratio (vs kelas terbesar)")
    axes[2].set_title("Rasio Ketidakseimbangan Kelas")
    for i, r in enumerate(ratios):
        axes[2].text(r + 0.05, i, f"{r:.2f}x", va="center", fontsize=9)
    axes[2].axvline(x=1.0, color="green", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(output_dir, f"data_distribution_classes_{level_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ===== FIGURE 2: Distribusi fitur-fitur representatif =====
    # Pilih fitur yang paling diskriminatif menggunakan ANOVA F-test
    X_feat = df[feature_cols].values
    y_feat = df[class_col].values
    try:
        selector = SelectKBest(f_classif, k=min(8, len(feature_cols)))
        selector.fit(X_feat, y_feat)
        top_indices = np.argsort(selector.scores_)[::-1][:8]
        top_feat_names = [feature_cols[i] for i in top_indices]
    except Exception:
        top_feat_names = feature_cols[:8]

    n_top = len(top_feat_names)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, feat_name in enumerate(top_feat_names):
        if i >= 8:
            break
        data_per_class = []
        for cls in class_names:
            data_per_class.append(df[df[class_col] == cls][feat_name].values)

        parts = axes[i].violinplot(data_per_class, positions=range(n_classes), showmeans=True, showmedians=True)
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[j])
            pc.set_alpha(0.7)
        parts["cmeans"].set_color("red")
        parts["cmedians"].set_color("blue")

        axes[i].set_xticks(range(n_classes))
        axes[i].set_xticklabels(class_names, rotation=25, ha="right", fontsize=8)
        axes[i].set_title(feat_name, fontsize=10)
        axes[i].grid(True, alpha=0.3, axis="y")

    # Kosongkan subplot yang tidak terpakai
    for i in range(n_top, 8):
        axes[i].set_visible(False)

    fig.suptitle(f"Distribusi Top-{n_top} Fitur Paling Diskriminatif {level_name}", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, f"data_distribution_features_{level_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ===== FIGURE 3: Korelasi SEMUA fitur + PCA =====
    # Gunakan SEMUA fitur yang tersedia (bukan top-12 hardcoded)
    # → Saat FK nonaktif  : 11 fitur
    # → Saat FK aktif     : 14 fitur (+ Back_Azimuth, Slowness, Beam_Power)
    # → Jika ada tambahan : otomatis ikut tampil
    all_feat_cols = feature_cols   # tidak di-slice, pakai semua

    n_all_feats = len(all_feat_cols)

    # Ukuran figur disesuaikan otomatis dengan jumlah fitur
    # agar label tidak bertabrakan dan anotasi masih terbaca
    _cell    = 0.65                        # inch per sel heatmap
    _hm_side = _cell * n_all_feats + 2.0  # sisi persegi heatmap
    _hm_side = max(_hm_side, 7.0)         # minimal 7 inch
    _fig_w   = _hm_side + 6.5             # +6.5 untuk panel PCA di sebelah kanan

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(_fig_w, _hm_side),
        gridspec_kw={"width_ratios": [_hm_side, 5.5]},  # heatmap lebih lebar
    )

    # 3a. Heatmap korelasi — SEMUA fitur (lower triangle)
    corr_matrix = df[all_feat_cols].corr()
    mask_triu   = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Ukuran font anotasi menyesuaikan jumlah fitur
    annot_fs = max(5, 9 - n_all_feats // 3)

    sns.heatmap(
        corr_matrix,
        mask=mask_triu,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax1,
        xticklabels=[n[:18] for n in all_feat_cols],   # maks 18 karakter
        yticklabels=[n[:18] for n in all_feat_cols],
        annot_kws={"fontsize": annot_fs},
        linewidths=0.4,
        linecolor="white",
        square=True,                    # sel persegi agar konsisten
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    ax1.set_title(
        f"Korelasi Semua Fitur ({n_all_feats} fitur) {level_name}",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax1.tick_params(axis="x", labelsize=8, rotation=45)
    ax1.tick_params(axis="y", labelsize=8, rotation=0)

    # 3b. PCA 2D scatter (tidak berubah, hanya dipindahkan ke ax2)
    scaler_pca   = StandardScaler()
    X_scaled_pca = scaler_pca.fit_transform(X_feat)
    pca          = PCA(n_components=2, random_state=SEED)
    X_pca        = pca.fit_transform(X_scaled_pca)

    for i, cls in enumerate(class_names):
        mask_cls = y_feat == cls
        ax2.scatter(
            X_pca[mask_cls, 0], X_pca[mask_cls, 1],
            c=[colors[i]], label=cls, alpha=0.6,
            edgecolors="white", linewidth=0.5, s=40,
        )
    ax2.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=11)
    ax2.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=11)
    ax2.set_title(f"PCA 2D Projection {level_name}", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f"data_distribution_corr_pca_{level_name.lower().replace(' ', '_')}.png",
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Print ringkasan statistik
    print(f"\n  --- Ringkasan Data {level_name} ---")
    print(f"    Total sampel       : {len(df)}")
    print(f"    Jumlah fitur       : {len(feature_cols)}")
    print(f"    Jumlah kelas       : {n_classes}")
    for cls, cnt in class_counts.items():
        print(f"    {cls:20s}: {cnt:4d} ({100*cnt/len(df):.1f}%)")
    imbalance = class_counts.max() / class_counts.min()
    print(f"    Rasio imbalance    : {imbalance:.2f}x")
    print(f"    PCA variance ratio : PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

# ===========================================================================
# PLOT: SPLIT MODEL COMPARISON (SVM vs XGBoost per Kelas)
# ===========================================================================

def plot_split_model_comparison(y_test, svm_pred, xgb_pred, class_names, test_size, output_plots,
                                subset_label="Test"):
    """
    Membuat figure 4-subplot (Precision | Recall | F1-Score | Accuracy) yang
    membandingkan SVM vs XGBoost per kelas.

    Parameter
    ---------
    y_test       : array-like   — Label aktual (integer encoded). Meski nama
                   parameter "y_test", fungsi ini generik — bisa dipakai untuk
                   evaluasi set apa saja (test ATAU training), tinggal isi
                   subset_label sesuai.
    svm_pred     : array-like   — Prediksi SVM.
    xgb_pred     : array-like   — Prediksi XGBoost.
    class_names  : list[str]    — Nama kelas dari LabelEncoder.
    test_size    : float        — Proporsi test set (misal 0.4 → "Split 60/40"),
                   dipakai untuk label "Split X/Y" di judul & nama file.
    output_plots : str          — Folder tujuan simpan gambar.
    subset_label : str          — "Test" (default) atau "Train" — dipakai di
                   judul plot dan sebagai infix nama file supaya tidak
                   bentrok antara versi test dan train.

    Output
    ------
    split_model_comparison_{train_pct}_{test_pct}.png              (subset_label="Test")
    split_model_comparison_train_{train_pct}_{test_pct}.png        (subset_label="Train")
    """
    _display_map = {
        "NonEvent":   "Gempa Bumi",
        "Multiphase": "MP",
        "Rockfall":   "RF",
        "VTB":        "VTB",
    }

    # ── Hitung metrik per kelas ──────────────────────────────────────────
    svm_prec, svm_rec, svm_f1, _ = precision_recall_fscore_support(
        y_test, svm_pred, average=None, zero_division=0
    )
    xgb_prec, xgb_rec, xgb_f1, _ = precision_recall_fscore_support(
        y_test, xgb_pred, average=None, zero_division=0
    )

    # Accuracy per kelas = diagonal CM / jumlah aktual per kelas
    svm_cm  = confusion_matrix(y_test, svm_pred)
    xgb_cm  = confusion_matrix(y_test, xgb_pred)
    svm_acc = svm_cm.diagonal() / svm_cm.sum(axis=1)
    xgb_acc = xgb_cm.diagonal() / xgb_cm.sum(axis=1)

    # ── Label & judul ────────────────────────────────────────────────────
    n_classes  = len(class_names)
    x          = np.arange(n_classes)
    width      = 0.35
    train_pct  = int(round((1 - test_size) * 100))
    test_pct   = int(round(test_size * 100))

    xlabel_list  = []
    legend_parts = []
    for i, cn in enumerate(class_names):
        short = _display_map.get(cn, cn)
        xlabel_list.append(f"Class {i}\n({short})")
        legend_parts.append(f"{i}: {short}")

    title_main = (
        f"Perbandingan SVM vs XGBoost per Kelas — Split {train_pct}/{test_pct} "
        f"({subset_label} Set)"
    )
    title_sub  = "(" + " | ".join(legend_parts) + ")"

    SVM_COLOR = "#4472C4"   # biru
    XGB_COLOR = "#ED7D31"   # oranye

    metrics = [
        ("Precision", svm_prec, xgb_prec),
        ("Recall",    svm_rec,  xgb_rec),
        ("F1-Score",  svm_f1,   xgb_f1),
        ("Accuracy",  svm_acc,  xgb_acc),   # ← subplot ke-4
    ]

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=False)
    fig.suptitle(f"{title_main}\n{title_sub}", fontsize=13, fontweight="bold", y=1.03)

    for ax, (metric_name, svm_vals, xgb_vals) in zip(axes, metrics):
        bars_svm = ax.bar(x - width / 2, svm_vals, width,
                          label="SVM", color=SVM_COLOR, zorder=3)
        bars_xgb = ax.bar(x + width / 2, xgb_vals, width,
                          label="XGBoost", color=XGB_COLOR, zorder=3)

        # Anotasi nilai di atas setiap bar
        for bar in bars_svm:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )
        for bar in bars_xgb:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

        ax.set_title(metric_name, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabel_list, fontsize=9)
        ax.set_ylim(0.0, 1.15)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.3, axis="y", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    file_infix = "" if subset_label.lower() == "test" else f"{subset_label.lower()}_"
    save_path = os.path.join(
        output_plots, f"split_model_comparison_{file_infix}{train_pct}_{test_pct}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot: {save_path}")


# ===========================================================================
# REPORT: EVALUATION REPORT (.txt)
# ===========================================================================

def save_evaluation_report(
    y_test, svm_pred, xgb_pred,
    svm_cv_scores, xgb_cv_scores,
    svm_best_params, xgb_best_params,
    class_names, test_size, output_plots,
):
    """
    Menyimpan laporan evaluasi lengkap dalam format .txt.

    Isi Laporan
    -----------
    - Header: tanggal, jumlah sampel, split ratio
    - Per model (SVM & XGBoost):
        * Akurasi keseluruhan
        * CV Accuracy mean ± std
        * Best hyperparameters
        * Classification report (precision / recall / F1 / support)
        * Confusion matrix
        * Akurasi per kelas
    - Footer

    Parameter
    ---------
    y_test           : array-like   — Label aktual.
    svm_pred         : array-like   — Prediksi SVM.
    xgb_pred         : array-like   — Prediksi XGBoost.
    svm_cv_scores    : array-like   — Skor CV SVM (dari cross_val_score).
    xgb_cv_scores    : array-like   — Skor CV XGBoost.
    svm_best_params  : dict         — Best params GridSearchCV SVM.
    xgb_best_params  : dict         — Best params GridSearchCV XGBoost.
    class_names      : list[str]    — Nama kelas dari LabelEncoder.
    test_size        : float        — Proporsi test set.
    output_plots     : str          — Folder tujuan simpan file.

    Output
    ------
    evaluation_report_{train_pct}_{test_pct}.txt
    """
    train_pct = int(round((1 - test_size) * 100))
    test_pct  = int(round(test_size * 100))
    sep       = "=" * 70
    sep_thin  = "-" * 70

    lines = []
    lines += [
        sep,
        "  SEISMIC EVENT CLASSIFICATION — EVALUATION REPORT",
        f"  Split: {train_pct}% Train / {test_pct}% Test",
        sep,
        f"  Generated   : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Test samples: {len(y_test)}",
        f"  Classes     : {list(class_names)}",
        "",
    ]

    model_runs = [
        ("SVM-RBF",  svm_pred,  svm_cv_scores,  svm_best_params),
        ("XGBoost",  xgb_pred,  xgb_cv_scores,  xgb_best_params),
    ]

    for model_name, y_pred, cv_scores, best_params in model_runs:
        cm        = confusion_matrix(y_test, y_pred)
        class_acc = cm.diagonal() / cm.sum(axis=1) * 100

        lines += [
            sep,
            f"  MODEL: {model_name}",
            sep,
            "",
            f"  Overall Accuracy  : {accuracy_score(y_test, y_pred):.4f}",
            f"  CV Accuracy (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
            f"  Best Params       : {best_params}",
            "",
            "  Classification Report:",
            "  " + sep_thin,
        ]

        report_lines = classification_report(
            y_test, y_pred, target_names=class_names, digits=3
        ).splitlines()
        lines += ["  " + ln for ln in report_lines]

        lines += [
            "",
            "  Confusion Matrix:",
            "  " + sep_thin,
        ]

        n_classes = len(class_names)

        # Header kolom
        col_w = max(len(cn) for cn in class_names) + 2
        header_row = "  " + " " * (col_w + 2) + "  ".join(
            f"{cn:>{col_w}}" for cn in class_names
        )
        lines.append(header_row)
        lines.append("  " + " " * (col_w + 2) + sep_thin[:n_classes * (col_w + 2)])

        for i, row in enumerate(cm):
            row_str = (
                "  "
                + f"{class_names[i]:<{col_w}}| "
                + "  ".join(f"{v:>{col_w}d}" for v in row)
            )
            lines.append(row_str)

        lines += [
            "",
            "  Per-Class Accuracy:",
            "  " + sep_thin,
        ]
        for cn, acc in zip(class_names, class_acc):
            lines.append(f"    {cn:<20s}: {acc:.2f}%")

        lines.append("")

    lines += [sep, "  END OF REPORT", sep]

    report_path = os.path.join(
        output_plots, f"evaluation_report_{train_pct}_{test_pct}.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"   Report: {report_path}")


def save_confusion_matrix_examples(
    df, X_test, y_pred, le, data_dir, output_dir, tag, model_name,
    n_examples=3, station=None, random_state=42,
    paz=None, prefilter=None, target_fs=None, fmin=None, fmax=None,
):
    """
    Masukan dari dosen pembimbing: "data hasil testing minta ditampilkan
    prediksi sinyalnya kayak apa, kalau dilihat matriks confusion" —
    fungsi ini menampilkan confusion matrix bukan cuma sebagai angka, tapi
    sebagai SINYAL SUNGGUHAN: untuk tiap sel (label ASLI x label PREDIKSI)
    dari hasil model pada TEST SET, simpan beberapa contoh waveform+spektrum
    event yang masuk ke sel itu — termasuk sel misklasifikasi (mis. label
    asli VTB tapi diprediksi Rockfall) — supaya bisa dicek visual kenapa
    model salah/benar mengklasifikasikan.

    Cara kerja
    ----------
    1. df.loc[X_test.index] memetakan baris X_test kembali ke Event & Label
       ASLI (index asli tetap terjaga pandas setelah train_test_split).
    2. le.inverse_transform(y_pred) memberi label PREDIKSI model.
    3. Untuk tiap kombinasi (label asli, label prediksi) yang benar-benar
       muncul di test set, ambil hingga n_examples event secara acak, cari
       file .mseed stasiun referensi-nya (di subfolder LABEL ASLI, karena
       struktur folder data_dir mengikuti label asli/ground truth).
    4. Preprocess dengan preprocess_stream_per_event() — SAMA PERSIS dengan
       pipeline fitur — lalu simpan waveform+spektrum per event.
    5. Buat 1 figure ringkasan berbentuk grid confusion matrix (baris=label
       asli, kolom=label prediksi), tiap sel diisi 1 waveform representatif
       — sel diagonal (prediksi benar) diberi bingkai hijau, sel non-diagonal
       (misklasifikasi) diberi bingkai merah.

    Struktur Output
    ---------------
    output_dir/confusion_examples_{tag}/{model_name}/
    ├── True_{label_asli}__Pred_{label_prediksi}/{event}.png   (hingga n_examples)
    └── confusion_matrix_waveforms_{model_name}_{tag}.png       (grid ringkasan)

    Parameter
    ---------
    df         : pd.DataFrame — dataset fitur lengkap (kolom Event & Label).
    X_test     : pd.DataFrame — subset fitur test hasil train_test_split
                 (index-nya harus tetap bagian dari df.index).
    y_pred     : array-like   — prediksi model (integer encoded) untuk X_test,
                 urutan harus SAMA dengan X_test (mis. model.predict(X_test)).
    le         : LabelEncoder — dipakai untuk urutan & nama kelas.
    data_dir   : str  — folder data mentah .mseed (struktur subfolder label ASLI).
    output_dir : str  — folder simpan plot.
    tag        : str  — nama split, dipakai untuk penamaan folder/file output.
    model_name : str  — nama model (mis. "SVM", "XGBoost"), dipakai di path/judul.
    n_examples : int  — maksimal jumlah contoh event disimpan per sel.
    station    : str | None — kode stasiun yang diplot. Default: REFERENCE_STATION.
    """
    if station is None:
        station = REFERENCE_STATION
    if not data_dir or not os.path.isdir(data_dir):
        print(f"   [WARN] data_dir tidak valid ({data_dir}) — contoh confusion matrix dilewati.")
        return

    if paz       is None: paz       = _DEFAULT_PAZ
    if prefilter is None: prefilter = _PREFILTER
    if target_fs is None: target_fs = _TARGET_FS
    if fmin      is None: fmin      = _FMIN
    if fmax      is None: fmax      = _FMAX

    class_names = list(le.classes_)
    n_cls       = len(class_names)
    rng         = np.random.RandomState(random_state)

    test_info = df.loc[X_test.index, ["Event", "Label"]].copy()
    test_info["Pred_Label"] = le.inverse_transform(np.asarray(y_pred))

    print(f"\n  Menyimpan contoh waveform per sel confusion matrix "
          f"[{model_name}, {tag}]...")

    model_dir = os.path.join(output_dir, f"confusion_examples_{tag}", model_name)
    os.makedirs(model_dir, exist_ok=True)

    # (true_cls, pred_cls) -> (event_id, data, sr)  — dipakai untuk grid ringkasan
    grid_examples = {}
    total_saved, total_failed = 0, 0

    for true_cls in class_names:
        for pred_cls in class_names:
            subset = test_info[
                (test_info["Label"] == true_cls) & (test_info["Pred_Label"] == pred_cls)
            ]
            if subset.empty:
                continue

            events = subset["Event"].unique().tolist()
            rng.shuffle(events)
            chosen = events[:n_examples]

            cell_dir = os.path.join(model_dir, f"True_{true_cls}__Pred_{pred_cls}")

            for i, event_id in enumerate(chosen):
                direct = os.path.join(data_dir, true_cls, f"{event_id}_{station}.mseed")
                if os.path.exists(direct):
                    filepath = direct
                else:
                    matches = glob.glob(os.path.join(data_dir, true_cls, f"{event_id}*{station}*.mseed"))
                    filepath = matches[0] if matches else None

                if filepath is None:
                    print(f"   [WARN] File stasiun {station} tidak ditemukan untuk "
                          f"event {event_id} (True={true_cls}, Pred={pred_cls}).")
                    total_failed += 1
                    continue

                try:
                    st_raw = read(filepath)
                    st = preprocess_stream_per_event(
                        st_raw, paz=paz, prefilter=prefilter,
                        target_fs=target_fs, fmin=fmin, fmax=fmax,
                    )
                    if len(st) == 0:
                        raise ValueError("preprocessing menghasilkan stream kosong")
                    tr   = st[0]
                    data = tr.data.astype(np.float64)
                    sr   = tr.stats.sampling_rate
                except Exception as e:
                    print(f"   [WARN] Gagal memuat waveform {event_id}: {e}")
                    total_failed += 1
                    continue

                correct = (true_cls == pred_cls)
                border_color = "#4CAF50" if correct else "#E53935"

                t = np.arange(len(data)) / sr
                fig, (ax_wave, ax_spec) = plt.subplots(
                    1, 2, figsize=(13, 3.2), gridspec_kw={"width_ratios": [3, 1]}
                )
                ax_wave.plot(t, data, color=border_color, linewidth=0.7, alpha=0.9)
                status = "BENAR" if correct else "SALAH KLASIFIKASI"
                ax_wave.set_title(
                    f"Event: {event_id}  |  True: {true_cls}  →  Pred: {pred_cls}  ({status})",
                    fontsize=10, fontweight="bold", loc="left", pad=20,
                )
                ax_wave.set_xlabel("Waktu (detik)", fontsize=8)
                ax_wave.set_ylabel("Amplitudo (m/s)", fontsize=8)
                ax_wave.tick_params(labelsize=7)
                ax_wave.grid(True, alpha=0.3)
                ax_wave.margins(x=0)
                for spine in ax_wave.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2)

                fft_mag = np.abs(np.fft.rfft(data))
                freqs   = np.fft.rfftfreq(len(data), d=1.0 / sr)
                mask    = freqs <= (fmax + 5)
                ax_spec.plot(freqs[mask], fft_mag[mask], color=border_color, linewidth=0.8, alpha=0.9)
                ax_spec.set_xlabel("Frekuensi (Hz)", fontsize=8)
                ax_spec.set_ylabel("|FFT|", fontsize=8)
                ax_spec.set_title("Spektrum", fontsize=9)
                ax_spec.tick_params(labelsize=7)
                ax_spec.grid(True, alpha=0.3)
                ax_spec.margins(x=0)

                plt.tight_layout()
                os.makedirs(cell_dir, exist_ok=True)
                safe_name = "".join(c for c in str(event_id) if c.isalnum() or c in "_-.")
                fig_path = os.path.join(cell_dir, f"{safe_name}.png")
                plt.savefig(fig_path, dpi=120, bbox_inches="tight")
                plt.close()
                total_saved += 1

                if i == 0:
                    grid_examples[(true_cls, pred_cls)] = (event_id, data, sr, len(events))

    print(f"   ✅ Contoh per sel tersimpan: {total_saved} berhasil, "
          f"{total_failed} gagal → {model_dir}")

    # ── Grid ringkasan: confusion matrix "dilihat sebagai sinyal" ─────────
    fig, axes = plt.subplots(n_cls, n_cls, figsize=(4.2 * n_cls, 2.6 * n_cls))
    if n_cls == 1:
        axes = np.array([[axes]])

    for i, true_cls in enumerate(class_names):
        for j, pred_cls in enumerate(class_names):
            ax = axes[i, j]
            key = (true_cls, pred_cls)

            if key not in grid_examples:
                ax.text(0.5, 0.5, "Tidak ada\ncontoh", ha="center", va="center",
                        fontsize=9, color="gray", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#CCCCCC")
                continue

            event_id, data, sr, n_avail = grid_examples[key]
            t = np.arange(len(data)) / sr
            correct = (true_cls == pred_cls)
            color = "#4CAF50" if correct else "#E53935"

            ax.plot(t, data, color=color, linewidth=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"n={n_avail}", fontsize=8, color=color, pad=2)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2 if correct else 1.5)

            if i == n_cls - 1:
                ax.set_xlabel(pred_cls, fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(true_cls, fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Confusion Matrix sebagai Waveform — {model_name} [{tag}]\n"
        f"Baris = Label Asli, Kolom = Label Prediksi "
        f"(hijau = benar, merah = salah klasifikasi)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.text(0.5, -0.01,
              "Kolom → Label Prediksi   |   Baris ↓ Label Asli",
              ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    grid_path = os.path.join(
        output_dir, f"confusion_matrix_waveforms_{model_name}_{tag}.png"
    )
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Grid ringkasan: {grid_path}")


def run_evaluation(output_csv, model_output, output_plots, test_size=None, random_state=42,
                   data_dir=None, tag=None):
    """
    Menjalankan evaluasi lengkap dan menghasilkan semua plot visualisasi.

    Plot yang Dihasilkan
    --------------------
    Evaluasi pada TEST set (data yang tidak dilihat saat training):
    - class_dist.png    : Distribusi kelas dataset (class imbalance check)
    - cm_SVM.png        : Confusion Matrix SVM
    - cm_XGBoost.png    : Confusion Matrix XGBoost
    - roc_SVM.png       : ROC Curve per kelas — SVM
    - roc_XGBoost.png   : ROC Curve per kelas — XGBoost
    - lc_SVM.png        : Learning Curve — SVM (bias-variance diagnosis)
    - lc_XGBoost.png    : Learning Curve — XGBoost
    - split_model_comparison_{tr}_{te}.png : Precision/Recall/F1/Acc SVM vs XGB

    Hasil klasifikasi dan misklasifikasi per event:
    - classification_results_{model}_{tag}.csv
      → seluruh event test, true label, predicted label, status, confidence
        dan probabilitas tiap kelas.
    - misclassification_{model}_{tag}.csv
      → hanya event yang salah diklasifikasikan.
    - misclassification_summary_{model}_{tag}.txt
      → ringkasan pasangan kelas yang tertukar dan error per kelas.

    Evaluasi pada TRAINING set (data yang DIPAKAI untuk training — untuk
    dibandingkan dengan hasil test set di atas, mendiagnosis overfitting):
    - cm_train_SVM.png, cm_train_XGBoost.png     : Confusion Matrix
    - roc_train_SVM.png, roc_train_XGBoost.png   : ROC Curve per kelas
    - split_model_comparison_train_{tr}_{te}.png : Precision/Recall/F1/Acc

    Confusion matrix sebagai SINYAL (hanya jika data_dir diisi — masukan
    dosen pembimbing: "tampilkan prediksi sinyal kayak apa kalau dilihat
    matriks confusion"). Lihat save_confusion_matrix_examples():
    - confusion_examples_{tag}/{model}/True_{asli}__Pred_{prediksi}/{event}.png
      → contoh waveform+spektrum per sel confusion matrix test set,
        termasuk sel misklasifikasi.
    - confusion_matrix_waveforms_{model}_{tag}.png
      → 1 figure grid (baris=label asli, kolom=label prediksi) berisi
        1 waveform representatif per sel (hijau=benar, merah=salah).

    Metrik yang Dilaporkan
    ----------------------
    - Classification report (precision, recall, F1 per kelas + macro avg),
      untuk TEST set maupun TRAINING set
    - Confusion matrix numerik
    - Akurasi per kelas (diagonal CM / jumlah sampel aktual per kelas)
    - Cross-validation accuracy (5-fold, mean ± std)

    Parameter
    ---------
    output_csv   : str  — Path CSV dataset.
    model_output : str  — Folder berisi seismic_models.joblib.
    output_plots : str  — Folder penyimpanan semua plot.
    random_state : int  — Seed untuk reproducibility.
    data_dir     : str | None — Folder data mentah .mseed (struktur subfolder
                   label). Jika diisi, membuat contoh waveform per sel
                   confusion matrix (lihat di atas). Jika None, langkah ini
                   dilewati.
    """
    print()
    print("=" * 60)
    print("TAHAP 3: EVALUASI & VISUALISASI")
    print("=" * 60)

    os.makedirs(output_plots, exist_ok=True)

    model_path = os.path.join(model_output, "seismic_models.joblib")
    if not os.path.exists(model_path):
        print(f"  [ERROR] Model tidak ditemukan: {model_path}")
        print("  Jalankan --mode train terlebih dahulu.")
        sys.exit(1)

    saved     = joblib.load(model_path)
    # Ambil test_size dari file model jika tidak di-pass eksplisit
    if test_size is None:
        test_size = saved.get("test_size", TEST_SIZE)
    if tag is None:
        train_pct = int(round((1 - test_size) * 100))
        test_pct  = int(round(test_size * 100))
        tag = f"{train_pct}_{test_pct}"
    le          = saved["label_encoder"]
    num_cols    = saved["feature_cols"]
    svm_best    = saved["svm_best"]
    xgb_best    = saved["xgb_best"]
    class_names = le.classes_

    df     = pd.read_csv(output_csv)
    X_full = df[num_cols].copy()
    y_full = le.transform(df["Label"])

    X_test = saved.get("X_test")
    y_test = saved.get("y_test")

    if X_test is None or y_test is None:
        _, X_test, _, y_test = train_test_split(
            X_full, y_full, test_size=TEST_SIZE, random_state=random_state, stratify=y_full
        )

    X_train = saved.get("X_train")
    y_train = saved.get("y_train")

    if X_train is None or y_train is None:
        # Model lama (belum menyimpan X_train/y_train) — rekonstruksi dari
        # X_full dengan membuang index yang sudah dipakai X_test, supaya
        # tetap konsisten (tanpa perlu tahu urutan blind holdout asli).
        print("  [WARN] X_train/y_train tidak tersimpan di model (versi lama) — "
              "direkonstruksi dari X_full minus X_test.")
        train_mask = ~X_full.index.isin(X_test.index)
        X_train = X_full.loc[train_mask]
        y_train = y_full[train_mask]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    print(f"  Kelas  : {list(class_names)}")
    print(f"  Test N : {len(y_test)}")
    print("  Output tambahan: classification results + misclassification per event")
    print()

    print("  Membuat plot distribusi kelas...")
    plot_class_distribution(
        df["Label"], list(class_names),
        save_path=os.path.join(output_plots, "class_dist.png")
    )

    # Kumpulkan prediksi & hasil CV per model untuk fungsi baru
    _preds       = {}
    _preds_train = {}
    _cv_scores   = {}
    _bp          = {}

    for name, model in [("SVM", svm_best), ("XGBoost", xgb_best)]:
        print(f"\n {'='*35}\n Evaluasi {name}\n {'='*35}")

        y_pred  = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        # ---------------------------------------------------------------
        # CLASSIFICATION RESULT + MISCLASSIFICATION ANALYSIS
        # ---------------------------------------------------------------
        save_classification_and_misclassification(
            df=df,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_score=y_score,
            le=le,
            model_name=name,
            output_plots=output_plots,
            tag=tag,
        )

        # Simpan untuk dipakai fungsi lain
        _preds[name]     = y_pred
        _cv_scores[name] = cross_val_score(model, X_full, y_full, cv=cv, scoring="accuracy")

        # Ambil best_params (hanya step 'model' dari pipeline)
        try:
            _bp[name] = {
                k: v for k, v in model.get_params().items()
                if k.startswith("model__") and v is not None
            }
        except Exception:
            _bp[name] = {}

        print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

        cm        = confusion_matrix(y_test, y_pred)
        class_acc = cm.diagonal() / cm.sum(axis=1) * 100
        acc_df    = pd.DataFrame({"Class": class_names, "Accuracy (%)": np.round(class_acc, 2)})
        print(f" Akurasi per kelas ({name}):")
        print(acc_df.to_string(index=False))
        print(f"\n CV Accuracy: {_cv_scores[name].mean():.3f} ± {_cv_scores[name].std():.3f}")

        plot_confusion_matrix(
            y_test, y_pred, class_names,
            title=f"Confusion Matrix — {name}",
            save_path=os.path.join(output_plots, f"cm_{name}.png"),
        )
        plot_roc_curve(
            y_test, y_score, class_names,
            title=f"ROC Curve — {name}",
            save_path=os.path.join(output_plots, f"roc_{name}.png"),
        )
        plot_learning_curve(
            model, X_full, y_full,
            title=f"Learning Curve — {name}",
            cv=cv,
            save_path=os.path.join(output_plots, f"lc_{name}.png"),
        )
        plot_learning_curve_2(
            model,
            X_full,
            y_full,
            name,
            class_names,
            output_plots,
        )

        # ── Evaluasi TRAINING set (dibandingkan dengan test set di atas —
        #    gap besar antara akurasi train vs test = indikasi overfitting,
        #    melengkapi learning curve yang sudah ada) ────────────────────
        print(f"\n {'-'*35}\n Evaluasi {name} — TRAINING SET\n {'-'*35}")

        y_pred_train  = model.predict(X_train)
        y_score_train = model.predict_proba(X_train)
        _preds_train[name] = y_pred_train

        print(classification_report(y_train, y_pred_train, target_names=class_names, digits=3))

        cm_train        = confusion_matrix(y_train, y_pred_train)
        class_acc_train = cm_train.diagonal() / cm_train.sum(axis=1) * 100
        acc_df_train    = pd.DataFrame({"Class": class_names, "Accuracy (%)": np.round(class_acc_train, 2)})
        print(f" Akurasi per kelas ({name}, training):")
        print(acc_df_train.to_string(index=False))

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc  = accuracy_score(y_test, y_pred)
        gap       = train_acc - test_acc
        gap_note  = "⚠️  gap besar, kemungkinan overfitting" if gap > 0.1 else "✅ gap wajar"
        print(f" Accuracy — Training: {train_acc:.3f}  |  Test: {test_acc:.3f}  |  "
              f"Gap: {gap:+.3f}  {gap_note}")

        plot_confusion_matrix(
            y_train, y_pred_train, class_names,
            title=f"Confusion Matrix (Training) — {name}",
            save_path=os.path.join(output_plots, f"cm_train_{name}.png"),
        )
        plot_roc_curve(
            y_train, y_score_train, class_names,
            title=f"ROC Curve (Training) — {name}",
            save_path=os.path.join(output_plots, f"roc_train_{name}.png"),
        )

        # ── Masukan dosen pembimbing: tampilkan confusion matrix sebagai
        #    sinyal (waveform), bukan cuma angka — termasuk sel misklasifikasi.
        if data_dir is not None:
            save_confusion_matrix_examples(
                df=df, X_test=X_test, y_pred=y_pred, le=le,
                data_dir=data_dir, output_dir=output_plots, tag=tag,
                model_name=name, n_examples=3,
            )

    plot_data_distribution_2(df, num_cols, "Label", output_plots)

    # ── Tambahan baru ──────────────────────────────────────────────────────
    print("\n Membuat split model comparison chart (test set)...")
    plot_split_model_comparison(
        y_test,
        svm_pred=_preds["SVM"],
        xgb_pred=_preds["XGBoost"],
        class_names=class_names,
        test_size=test_size,
        output_plots=output_plots,
        subset_label="Test",
    )

    print(" Membuat split model comparison chart (training set)...")
    plot_split_model_comparison(
        y_train,
        svm_pred=_preds_train["SVM"],
        xgb_pred=_preds_train["XGBoost"],
        class_names=class_names,
        test_size=test_size,
        output_plots=output_plots,
        subset_label="Train",
    )

    print(" Menyimpan evaluation report (.txt)...")
    save_evaluation_report(
        y_test,
        svm_pred=_preds["SVM"],
        xgb_pred=_preds["XGBoost"],
        svm_cv_scores=_cv_scores["SVM"],
        xgb_cv_scores=_cv_scores["XGBoost"],
        svm_best_params=_bp["SVM"],
        xgb_best_params=_bp["XGBoost"],
        class_names=class_names,
        test_size=test_size,
        output_plots=output_plots,
    )
    # ───────────────────────────────────────────────────────────────────────

    print(f"\n✅ Semua plot disimpan di: {output_plots}")


# ===========================================================================
# TAHAP 5 — BLIND TEST
# ===========================================================================

def save_blind_test_report(y_true, svm_pred, xgb_pred, class_names,
                            svm_cv_scores, xgb_cv_scores,
                            blind_size, output_dir):
    """
    Menyimpan laporan blind test ke file .txt.
    """
    sep      = "=" * 70
    sep_thin = "-" * 70
    n        = len(y_true)

    lines = [
        sep,
        "  SEISMIC EVENT CLASSIFICATION — BLIND TEST REPORT",
        f"  Blind test size : {blind_size*100:.0f}% ({n} sampel)",
        f"  Generated       : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Classes         : {list(class_names)}",
        sep, "",
    ]

    for model_name, y_pred, cv_scores in [
        ("SVM-RBF",  svm_pred,  svm_cv_scores),
        ("XGBoost",  xgb_pred,  xgb_cv_scores),
    ]:
        cm        = confusion_matrix(y_true, y_pred)
        class_acc = cm.diagonal() / cm.sum(axis=1) * 100

        lines += [
            sep,
            f"  MODEL: {model_name}",
            sep, "",
            f"  Overall Accuracy  : {accuracy_score(y_true, y_pred):.4f}",
            f"  F1-macro          : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
            f"  CV (train) mean   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
            "",
            "  Classification Report (Blind Set):",
            "  " + sep_thin,
        ]
        for ln in classification_report(
            y_true, y_pred, target_names=class_names, digits=3
        ).splitlines():
            lines.append("  " + ln)

        lines += ["", "  Confusion Matrix:", "  " + sep_thin]
        col_w = max(len(cn) for cn in class_names) + 2
        lines.append("  " + " " * (col_w + 2) + "  ".join(
            f"{cn:>{col_w}}" for cn in class_names))
        for i, row in enumerate(cm):
            lines.append(
                "  " + f"{class_names[i]:<{col_w}}| "
                + "  ".join(f"{v:>{col_w}d}" for v in row)
            )
        lines += ["", "  Per-Class Accuracy:", "  " + sep_thin]
        for cn, acc in zip(class_names, class_acc):
            lines.append(f"    {cn:<20s}: {acc:.2f}%")
        lines.append("")

    lines += [sep, "  END OF BLIND TEST REPORT", sep]

    report_path = os.path.join(output_dir, "blind_test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"   Report: {report_path}")


def plot_blind_test_results(y_true, svm_pred, xgb_pred,
                             svm_prob, xgb_prob,
                             class_names, output_dir):
    """
    Menghasilkan 4 plot blind test:
      1. Confusion Matrix SVM
      2. Confusion Matrix XGBoost
      3. ROC Curve per kelas — SVM
      4. ROC Curve per kelas — XGBoost
      5. Bar chart perbandingan F1 per kelas (SVM vs XGBoost)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Confusion Matrix ──────────────────────────────────────────────
    for name, y_pred in [("SVM", svm_pred), ("XGBoost", xgb_pred)]:
        plot_confusion_matrix(
            y_true, y_pred, class_names,
            title=f"Blind Test — Confusion Matrix {name}",
            save_path=os.path.join(output_dir, f"blind_cm_{name}.png"),
        )

    # ── ROC Curve ─────────────────────────────────────────────────────
    for name, y_score in [("SVM", svm_prob), ("XGBoost", xgb_prob)]:
        plot_roc_curve(
            y_true, y_score, class_names,
            title=f"Blind Test — ROC Curve {name}",
            save_path=os.path.join(output_dir, f"blind_roc_{name}.png"),
        )

    # ── F1 per kelas — SVM vs XGBoost ────────────────────────────────
    _, _, svm_f1, _ = precision_recall_fscore_support(
        y_true, svm_pred, average=None, zero_division=0)
    _, _, xgb_f1, _ = precision_recall_fscore_support(
        y_true, xgb_pred, average=None, zero_division=0)

    n_cls  = len(class_names)
    x      = np.arange(n_cls)
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, svm_f1, width,
                label="SVM", color="#4472C4", zorder=3)
    b2 = ax.bar(x + width/2, xgb_f1, width,
                label="XGBoost", color="#ED7D31", zorder=3)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Blind Test — F1-Score per Kelas (SVM vs XGBoost)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(output_dir, "blind_f1_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot: {path}")


def run_blind_test(output_csv, model_output, output_plots,
                   blind_size=0.15, random_state=42):
    """
    Menjalankan blind test: mengevaluasi model pada partisi data yang
    dipisahkan SEBELUM training dimulai (held-out set).

    Perbedaan dari run_evaluation()
    --------------------------------
    run_evaluation() menggunakan X_test dari train_test_split yang dipakai
    saat training. run_blind_test() menggunakan partisi terpisah (X_blind/
    y_blind) yang disisihkan oleh run_training() SEBELUM train/test split
    maupun GridSearchCV dijalankan, dan disimpan langsung di dalam
    seismic_models.joblib.

    PENTING — perbaikan dari versi sebelumnya
    ------------------------------------------
    Versi lama membuat train_test_split BARU dari CSV penuh dengan seed
    berbeda, dengan asumsi seed berbeda berarti data berbeda. Ini SALAH:
    partisi acak baru dari populasi yang sama hampir pasti tumpang tindih
    dengan data yang sudah dipakai untuk training (mis. ~60% dari sampel
    "blind" itu kemungkinan besar sudah pernah dilihat model). Sekarang
    blind set dibaca langsung dari partisi yang disisihkan run_training()
    di awal, sehingga dijamin belum pernah dilihat model dalam bentuk apapun.

    Alur
    ----
    1. Load model dari .joblib (termasuk X_blind/y_blind/X_trainval/y_trainval
       yang disimpan run_training()).
    2. Prediksi blind set dengan SVM & XGBoost.
    3. Hitung semua metrik (accuracy, F1, classification report,
       confusion matrix, ROC AUC), dan CV pembanding dihitung HANYA dari
       X_trainval/y_trainval (data non-blind) agar tidak ikut membocorkan
       blind set ke angka pembanding.
    4. Simpan plot dan laporan .txt ke output_plots/blind_test/.

    Parameter
    ---------
    output_csv   : str   — Tidak lagi dipakai untuk memisahkan blind set
                           (hanya untuk kompatibilitas argumen CLI);
                           blind set diambil dari .joblib.
    model_output : str   — Folder berisi seismic_models.joblib (harus hasil
                           run_training() versi baru yang menyimpan blind set).
    output_plots : str   — Folder induk output (subfolder blind_test/ dibuat otomatis).
    blind_size   : float — Hanya dipakai untuk validasi/peringatan jika berbeda
                           dari blind_size yang sudah dipakai saat training.
    random_state : int   — Hanya dipakai untuk validasi/peringatan jika berbeda
                           dari blind_seed yang sudah dipakai saat training.

    Output Files
    ------------
    blind_test/
    ├── blind_cm_SVM.png
    ├── blind_cm_XGBoost.png
    ├── blind_roc_SVM.png
    ├── blind_roc_XGBoost.png
    ├── blind_f1_comparison.png
    └── blind_test_report.txt

    Return
    ------
    dict — Metrik ringkasan: accuracy & F1-macro untuk SVM dan XGBoost.
    """
    print()
    print("=" * 60)
    print("TAHAP 5: BLIND TEST")
    print("=" * 60)

    blind_dir = os.path.join(output_plots, "blind_test")
    os.makedirs(blind_dir, exist_ok=True)

    # ── 1. Load model ─────────────────────────────────────────────────
    model_path = os.path.join(model_output, "seismic_models.joblib")
    if not os.path.exists(model_path):
        print(f"  [ERROR] Model tidak ditemukan: {model_path}")
        sys.exit(1)

    saved       = joblib.load(model_path)
    le          = saved["label_encoder"]
    num_cols    = saved["feature_cols"]
    svm_best    = saved["svm_best"]
    xgb_best    = saved["xgb_best"]
    class_names = le.classes_

    # ── 2. Ambil blind set yang SUDAH disisihkan run_training() ───────
    if saved.get("X_blind") is None or saved.get("y_blind") is None:
        print("  [ERROR] Model ini dilatih TANPA blind holdout")
        print("          (blind_size=0 saat training, atau versi model lama).")
        print("          Blind test dari potongan dataset yang sama tidak bisa")
        print("          dilakukan. Jalankan ulang --mode train dengan")
        print("          --blind_size > 0 jika memang butuh mode ini, atau")
        print("          pakai --mode blind_new dengan dataset TERPISAH.")
        sys.exit(1)

    X_blind    = saved["X_blind"]
    y_blind    = saved["y_blind"]
    X_trainval = saved.get("X_trainval")
    y_trainval = saved.get("y_trainval")
    saved_blind_size = saved.get("blind_size")
    saved_blind_seed = saved.get("blind_seed")

    if saved_blind_size is not None and abs(saved_blind_size - blind_size) > 1e-9:
        print(f"  [WARN] --blind_size={blind_size} diabaikan — model sudah "
              f"dilatih dengan blind_size={saved_blind_size} (partisi tetap).")
    if saved_blind_seed is not None and saved_blind_seed != random_state:
        print(f"  [WARN] --blind_seed={random_state} diabaikan — model sudah "
              f"dilatih dengan blind_seed={saved_blind_seed} (partisi tetap).")

    print(f"  Model         : {model_path}")
    print(f"  Blind set     : {len(X_blind)} sampel "
          f"(disisihkan saat training, blind_size={saved_blind_size}, "
          f"blind_seed={saved_blind_seed})")
    print(f"  Kelas         : {list(class_names)}")
    print()

    # ── 3. Prediksi ───────────────────────────────────────────────────
    svm_pred  = svm_best.predict(X_blind)
    svm_prob  = svm_best.predict_proba(X_blind)
    xgb_pred  = xgb_best.predict(X_blind)
    xgb_prob  = xgb_best.predict_proba(X_blind)

    # ── 4. Metrik konsol ──────────────────────────────────────────────
    # CV pembanding dihitung HANYA dari data non-blind (X_trainval/y_trainval)
    # agar blind set tidak ikut "bocor" ke angka CV pembanding ini.
    if X_trainval is None or y_trainval is None:
        print("  [WARN] X_trainval/y_trainval tidak tersedia di model — "
              "CV pembanding dilewati (kemungkinan model versi lama).")
        svm_cv = xgb_cv = np.array([np.nan])
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        svm_cv = cross_val_score(svm_best, X_trainval, y_trainval, cv=cv, scoring="accuracy")
        xgb_cv = cross_val_score(xgb_best, X_trainval, y_trainval, cv=cv, scoring="accuracy")

    results = {}
    for name, y_pred, cv_scores in [
        ("SVM",     svm_pred, svm_cv),
        ("XGBoost", xgb_pred, xgb_cv),
    ]:
        acc = accuracy_score(y_blind, y_pred)
        f1  = f1_score(y_blind, y_pred, average="macro", zero_division=0)
        results[name] = {"accuracy": acc, "f1_macro": f1}

        print(f"  {'='*35}")
        print(f"  {name} — Blind Test Results")
        print(f"  {'='*35}")
        print(f"  Accuracy   : {acc:.4f}")
        print(f"  F1-macro   : {f1:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print()
        print(classification_report(
            y_blind, y_pred, target_names=class_names, digits=3))

    # Perbandingan CV vs Blind
    print("  " + "=" * 45)
    print("  PERBANDINGAN: CV Score (Train) vs Blind Test")
    print("  " + "=" * 45)
    for name, cv_scores in [("SVM", svm_cv), ("XGBoost", xgb_cv)]:
        cv_mean = cv_scores.mean()
        blind_acc = results[name]["accuracy"]
        delta = blind_acc - cv_mean
        status = "✅ Generalisasi baik" if abs(delta) < 0.05 \
                 else ("⚠️  Kemungkinan overfit" if delta < -0.05
                       else "📈 Blind lebih tinggi dari CV")
        print(f"  {name:<10}: CV={cv_mean:.4f} | Blind={blind_acc:.4f} "
              f"| Δ={delta:+.4f}  {status}")
    print()

    # ── 5. Plot & laporan ─────────────────────────────────────────────
    print("  Membuat plot blind test...")
    plot_blind_test_results(
        y_blind, svm_pred, xgb_pred,
        svm_prob, xgb_prob,
        class_names, blind_dir,
    )

    print("  Menyimpan blind test report...")
    save_blind_test_report(
        y_blind, svm_pred, xgb_pred, class_names,
        svm_cv, xgb_cv,
        saved_blind_size if saved_blind_size is not None else blind_size,
        blind_dir,
    )

    print(f"\n✅ Blind test selesai. Output: {blind_dir}")
    return results

# ===========================================================================
# TAHAP 6 — BLIND TEST DATA BARU (EXTERNAL)
# ===========================================================================

def run_blind_test_new_data(
        pred_input,
        model_output,
        output_plots,
        station_coords,
        has_labels=False,
        already_cut=True,
        random_state=42,
):
    """
    Blind test menggunakan data .mseed BARU yang belum pernah dilihat model.

    Dua mode berdasarkan has_labels:
    ─────────────────────────────────────────────────────────
    has_labels=False (Unlabeled Blind):
        Data berada di satu flat folder tanpa subfolder label.
        Output: CSV prediksi + bar chart distribusi prediksi.
        Tidak ada metrik akurasi karena ground truth tidak diketahui.

    has_labels=True (Labeled Blind):
        Data berada di subfolder berlabel (sama seperti data training).
        Struktur: pred_input/Multiphase/*.mseed, pred_input/VTB/*.mseed, dst.
        Output: CSV prediksi + confusion matrix + ROC + F1 chart + report.
        Karena label diketahui, metrik evaluasi lengkap bisa dihitung.

    Parameter
    ─────────
    pred_input     : str   — Folder .mseed (flat atau berlabel subfolder).
    model_output   : str   — Folder berisi seismic_models.joblib.
    output_plots   : str   — Folder output (subfolder blind_new/ dibuat otomatis).
    station_coords : dict  — Koordinat stasiun {kode: (lat, lon)}.
    has_labels     : bool  — True jika data baru punya subfolder label.
    already_cut    : bool  — True jika data sudah dipotong per event.
    random_state   : int   — Seed untuk reproducibility.

    Output Files
    ─────────────
    blind_new/
    ├── blind_new_predictions.csv          ← selalu dihasilkan
    ├── blind_new_distribution.png         ← distribusi prediksi (unlabeled)
    ├── blind_new_cm_SVM.png               ← hanya jika has_labels=True
    ├── blind_new_cm_XGBoost.png           ← hanya jika has_labels=True
    ├── blind_new_roc_SVM.png              ← hanya jika has_labels=True
    ├── blind_new_roc_XGBoost.png          ← hanya jika has_labels=True
    ├── blind_new_f1_comparison.png        ← hanya jika has_labels=True
    └── blind_new_report.txt               ← hanya jika has_labels=True
    """
    import tempfile

    print()
    print("=" * 60)
    print("TAHAP 6: BLIND TEST — DATA BARU")
    print("=" * 60)
    print(f"  Input data   : {pred_input}")
    print(f"  Mode         : {'Labeled (evaluasi penuh)' if has_labels else 'Unlabeled (prediksi saja)'}")
    print(f"  already_cut  : {already_cut}")
    print()

    blind_dir = os.path.join(output_plots, "blind_new")
    os.makedirs(blind_dir, exist_ok=True)

    # ── 1. Load model ─────────────────────────────────────────────────
    model_path = os.path.join(model_output, "seismic_models.joblib")
    if not os.path.exists(model_path):
        print(f"  [ERROR] Model tidak ditemukan: {model_path}")
        sys.exit(1)

    saved       = joblib.load(model_path)
    le          = saved["label_encoder"]
    feature_cols = saved["feature_cols"]
    svm_best    = saved["svm_best"]
    xgb_best    = saved["xgb_best"]
    class_names = le.classes_

    print(f"  Model loaded  : {model_path}")
    print(f"  Kelas         : {list(class_names)}")
    print()

    # ── 2. Ekstraksi fitur data baru ──────────────────────────────────
    # Gunakan file CSV sementara di blind_dir
    tmp_csv = os.path.join(blind_dir, "_tmp_blind_new_features.csv")

    if has_labels:
        # Data berlabel → pakai run_feature_extraction() biasa
        # (subfolder = nama kelas, sama seperti training)
        df_feat = run_feature_extraction(
            data_dir       = pred_input,
            output_csv     = tmp_csv,
            station_coords = station_coords,
            valid_labels   = list(class_names),
        )
        y_true_str = df_feat["Label"].values

    else:
        # Data tidak berlabel → pakai run_feature_extraction_predict()
        df_feat = run_feature_extraction_predict(
            data_dir      = pred_input,
            output_csv    = tmp_csv,
            station_coords= station_coords,
            already_cut   = already_cut,
        )
        y_true_str = None

    if df_feat.empty:
        print("  [ERROR] Ekstraksi fitur gagal, tidak ada data.")
        sys.exit(1)

    # Align kolom dengan feature_cols model
    # (hindari mismatch jika ada kolom Event/Label)
    X_new = df_feat.reindex(columns=feature_cols, fill_value=0.0)
    print(f"  Fitur diekstrak: {len(X_new)} baris dari {X_new.shape[1]} fitur")
    print()

    # ── 3. Prediksi ───────────────────────────────────────────────────
    svm_pred  = svm_best.predict(X_new)
    svm_prob  = svm_best.predict_proba(X_new)
    xgb_pred  = xgb_best.predict(X_new)
    xgb_prob  = xgb_best.predict_proba(X_new)

    svm_label = le.inverse_transform(svm_pred)
    xgb_label = le.inverse_transform(xgb_pred)

    # Confidence = probabilitas kelas prediksi tertinggi
    svm_conf  = svm_prob.max(axis=1)
    xgb_conf  = xgb_prob.max(axis=1)

    # ── 4. Simpan CSV prediksi ────────────────────────────────────────
    out_df = df_feat[["Event"]].copy() if "Event" in df_feat.columns \
             else pd.DataFrame({"Event": [f"row_{i}" for i in range(len(X_new))]})

    if has_labels:
        out_df["True_Label"]      = y_true_str

    out_df["SVM_Pred"]        = svm_label
    out_df["SVM_Confidence"]  = svm_conf.round(4)
    out_df["XGB_Pred"]        = xgb_label
    out_df["XGB_Confidence"]  = xgb_conf.round(4)
    out_df["Agreement"]       = (svm_label == xgb_label)

    csv_out = os.path.join(blind_dir, "blind_new_predictions.csv")
    out_df.to_csv(csv_out, index=False)
    print(f"  ✅ Prediksi disimpan: {csv_out}")

    # ── 5. Tampilkan ringkasan prediksi ───────────────────────────────
    print()
    print("  Distribusi Prediksi:")
    print("  " + "-" * 35)
    for cls in class_names:
        n_svm = (svm_label == cls).sum()
        n_xgb = (xgb_label == cls).sum()
        print(f"  {cls:<15}: SVM={n_svm:4d}  XGBoost={n_xgb:4d}")
    agree_pct = out_df["Agreement"].mean() * 100
    print(f"\n  Model agreement: {agree_pct:.1f}%")
    print()

    # ── 6. Plot distribusi prediksi (untuk unlabeled) ─────────────────
    _plot_prediction_distribution(
        svm_label, xgb_label, class_names,
        save_path=os.path.join(blind_dir, "blind_new_distribution.png")
    )

    # ── 7. Evaluasi lengkap (jika has_labels=True) ────────────────────
    if has_labels:
        y_true_enc = le.transform(y_true_str)

        print("  " + "=" * 45)
        print("  EVALUASI BLIND TEST — DATA BERLABEL")
        print("  " + "=" * 45)

        results = {}
        for name, y_pred in [("SVM", svm_pred), ("XGBoost", xgb_pred)]:
            acc = accuracy_score(y_true_enc, y_pred)
            f1  = f1_score(y_true_enc, y_pred, average="macro", zero_division=0)
            results[name] = {"accuracy": acc, "f1_macro": f1}

            print(f"\n  {name}")
            print(f"  Accuracy : {acc:.4f}  |  F1-macro : {f1:.4f}")
            print(classification_report(
                y_true_enc, y_pred, target_names=class_names, digits=3))

        # Plot confusion matrix, ROC, F1 comparison
        plot_blind_test_results(
            y_true_enc, svm_pred, xgb_pred,
            svm_prob, xgb_prob, class_names,
            output_dir=blind_dir,
        )
        # Rename agar tidak timpa file dari run_blind_test()
        for old, new in [
            ("blind_cm_SVM.png",          "blind_new_cm_SVM.png"),
            ("blind_cm_XGBoost.png",      "blind_new_cm_XGBoost.png"),
            ("blind_roc_SVM.png",         "blind_new_roc_SVM.png"),
            ("blind_roc_XGBoost.png",     "blind_new_roc_XGBoost.png"),
            ("blind_f1_comparison.png",   "blind_new_f1_comparison.png"),
        ]:
            old_path = os.path.join(blind_dir, old)
            new_path = os.path.join(blind_dir, new)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)

        # Report teks
        _save_blind_new_report(
            y_true_enc, svm_pred, xgb_pred,
            class_names, blind_dir,
        )

    # Hapus CSV fitur sementara
    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)

    print(f"\n✅ Blind test data baru selesai. Output: {blind_dir}")
    return out_df


# ── Helper: plot distribusi prediksi ─────────────────────────────────────────
def _plot_prediction_distribution(svm_label, xgb_label, class_names,
                                   save_path):
    n_cls = len(class_names)
    x     = np.arange(n_cls)
    width = 0.35

    svm_counts = [(svm_label == c).sum() for c in class_names]
    xgb_counts = [(xgb_label == c).sum() for c in class_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, svm_counts, width,
                label="SVM", color="#4472C4", zorder=3)
    b2 = ax.bar(x + width/2, xgb_counts, width,
                label="XGBoost", color="#ED7D31", zorder=3)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{int(h)}",
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_ylabel("Jumlah Event Terprediksi", fontsize=11)
    ax.set_title("Blind Test Data Baru — Distribusi Prediksi (SVM vs XGBoost)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot: {save_path}")


# ── Helper: simpan laporan teks ───────────────────────────────────────────────
def _save_blind_new_report(y_true, svm_pred, xgb_pred, class_names, output_dir):
    sep = "=" * 70
    lines = [
        sep,
        "  BLIND TEST REPORT — DATA BARU (LABELED)",
        f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep, "",
    ]
    for name, y_pred in [("SVM-RBF", svm_pred), ("XGBoost", xgb_pred)]:
        lines += [
            sep, f"  MODEL: {name}", sep, "",
            f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}",
            f"  F1-macro : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
            "",
            "  Classification Report:",
        ]
        for ln in classification_report(
            y_true, y_pred, target_names=class_names, digits=3
        ).splitlines():
            lines.append("  " + ln)
        lines.append("")

    path = os.path.join(output_dir, "blind_new_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   Report: {path}")

# ===========================================================================
# TAHAP 7 — BLIND TEST DATA BARU — SEMUA SPLIT (LOOP OTOMATIS)
# ===========================================================================

def run_blind_new_all(
        pred_input,
        base_model_output,
        base_output_plots,
        station_coords,
        has_labels=False,
        already_cut=True,
        random_state=42,
):
    """
    Loop otomatis blind test data baru untuk SEMUA subfolder split
    yang ditemukan di dalam base_model_output.

    Subfolder yang valid harus:
      - Berisi file seismic_models.joblib
      - Namanya mengandung kata "split" (misal: split_60_40_no_fk)

    Alur
    ----
    1. Scan semua subfolder di base_model_output.
    2. Filter subfolder yang punya seismic_models.joblib.
    3. Untuk setiap split, panggil run_blind_test_new_data().
    4. Kumpulkan semua metrik dan cetak tabel perbandingan akhir.
    5. Simpan tabel perbandingan ke CSV dan PNG.

    Parameter
    ---------
    pred_input        : str   — Folder .mseed (flat atau berlabel).
    base_model_output : str   — Folder induk berisi subfolder split.
    base_output_plots : str   — Folder induk output plot.
    station_coords    : dict  — Koordinat stasiun {kode: (lat, lon)}.
    has_labels        : bool  — True jika data baru punya subfolder label.
    already_cut       : bool  — True jika data sudah dipotong per event.
    random_state      : int   — Seed reproducibility.

    Output Files
    ------------
    blind_new/
    ├── <split_tag>/               ← satu subfolder per split
    │   ├── blind_new_predictions.csv
    │   ├── blind_new_distribution.png
    │   ├── blind_new_cm_SVM.png       (jika has_labels)
    │   ├── blind_new_cm_XGBoost.png   (jika has_labels)
    │   ├── blind_new_roc_SVM.png      (jika has_labels)
    │   ├── blind_new_roc_XGBoost.png  (jika has_labels)
    │   ├── blind_new_f1_comparison.png(jika has_labels)
    │   └── blind_new_report.txt       (jika has_labels)
    ├── blind_all_summary.csv          ← ringkasan semua split
    └── blind_all_comparison.png       ← bar chart perbandingan
    """
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║     BLIND TEST DATA BARU — SEMUA SPLIT          ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Base model dir : {base_model_output}")
    print(f"  Input data     : {pred_input}")
    print(f"  Mode           : {'Labeled' if has_labels else 'Unlabeled'}")
    print()

    # ── 1. Scan subfolder yang punya seismic_models.joblib ────────────
    if not os.path.isdir(base_model_output):
        print(f"  [ERROR] Folder tidak ditemukan: {base_model_output}")
        sys.exit(1)

    valid_splits = []
    for sub in sorted(os.listdir(base_model_output)):
        sub_path   = os.path.join(base_model_output, sub)
        model_file = os.path.join(sub_path, "seismic_models.joblib")
        if os.path.isdir(sub_path) and os.path.exists(model_file):
            valid_splits.append((sub, sub_path))

    if not valid_splits:
        print(f"  [ERROR] Tidak ada subfolder dengan seismic_models.joblib "
              f"di: {base_model_output}")
        print("  Pastikan struktur folder:")
        print("    base_model_output/")
        print("    ├── split_60_40_no_fk/seismic_models.joblib")
        print("    ├── split_70_30_no_fk/seismic_models.joblib")
        print("    └── ...")
        sys.exit(1)

    print(f"  Split ditemukan ({len(valid_splits)}):")
    for tag, path in valid_splits:
        print(f"    ✔ {tag}")
    print()

    blind_root = os.path.join(base_output_plots, "blind_new")
    os.makedirs(blind_root, exist_ok=True)

    # ── 2. Loop tiap split ────────────────────────────────────────────
    all_summary = []

    for split_tag, model_dir in valid_splits:
        print()
        print("─" * 60)
        print(f"  SPLIT: {split_tag}")
        print("─" * 60)

        split_plot_dir = os.path.join(blind_root, split_tag)
        os.makedirs(split_plot_dir, exist_ok=True)

        # Patch: override output_plots agar run_blind_test_new_data()
        # menyimpan ke subfolder split yang benar
        try:
            out_df = _run_blind_single(
                pred_input     = pred_input,
                model_dir      = model_dir,
                split_plot_dir = split_plot_dir,
                station_coords = station_coords,
                has_labels     = has_labels,
                already_cut    = already_cut,
                random_state   = random_state,
            )
        except SystemExit:
            print(f"  [SKIP] {split_tag} gagal, lanjut ke split berikutnya.")
            continue

        # Kumpulkan metrik ringkasan
        summary_row = {"Split": split_tag}
        if has_labels and out_df is not None:
            # Baca metrik dari CSV prediksi
            summary_row["N_Samples"]      = len(out_df)
            summary_row["Agreement_%"]    = round(
                out_df["Agreement"].mean() * 100, 2)

            # Hitung akurasi dari kolom True_Label vs prediksi
            if "True_Label" in out_df.columns:
                svm_acc = (
                    out_df["True_Label"] == out_df["SVM_Pred"]
                ).mean()
                xgb_acc = (
                    out_df["True_Label"] == out_df["XGB_Pred"]
                ).mean()
                summary_row["SVM_Accuracy"]  = round(svm_acc, 4)
                summary_row["XGB_Accuracy"]  = round(xgb_acc, 4)
                summary_row["SVM_Conf_Mean"] = round(
                    out_df["SVM_Confidence"].mean(), 4)
                summary_row["XGB_Conf_Mean"] = round(
                    out_df["XGB_Confidence"].mean(), 4)
        else:
            if out_df is not None:
                summary_row["N_Samples"]   = len(out_df)
                summary_row["Agreement_%"] = round(
                    out_df["Agreement"].mean() * 100, 2)
                summary_row["SVM_Conf_Mean"] = round(
                    out_df["SVM_Confidence"].mean(), 4)
                summary_row["XGB_Conf_Mean"] = round(
                    out_df["XGB_Confidence"].mean(), 4)

        all_summary.append(summary_row)

    # ── 3. Tabel ringkasan semua split ────────────────────────────────
    if not all_summary:
        print("\n  [WARN] Tidak ada split yang berhasil diproses.")
        return

    df_summary = pd.DataFrame(all_summary)

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║        RINGKASAN BLIND TEST — SEMUA SPLIT       ║")
    print("╚══════════════════════════════════════════════════╝")
    print(df_summary.to_string(index=False))
    print()

    summary_csv = os.path.join(blind_root, "blind_all_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"  ✅ Summary CSV: {summary_csv}")

    # ── 4. Bar chart perbandingan antar split ─────────────────────────
    _plot_blind_all_comparison(df_summary, has_labels, blind_root)

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║          ✅ BLIND ALL SELESAI                    ║")
    print("╚══════════════════════════════════════════════════╝")
    return df_summary


# ── Helper: jalankan satu split (refactor dari run_blind_test_new_data) ───────
def _run_blind_single(pred_input, model_dir, split_plot_dir,
                      station_coords, has_labels, already_cut, random_state):
    """
    Versi internal run_blind_test_new_data() yang menerima path
    model_dir dan split_plot_dir secara eksplisit (bukan dibangun
    dari argumen induk). Dipakai oleh run_blind_new_all() agar
    setiap split menyimpan ke subfoldernya masing-masing.
    """
    import tempfile

    os.makedirs(split_plot_dir, exist_ok=True)

    # Load model
    model_path = os.path.join(model_dir, "seismic_models.joblib")
    saved        = joblib.load(model_path)
    le           = saved["label_encoder"]
    feature_cols = saved["feature_cols"]
    svm_best     = saved["svm_best"]
    xgb_best     = saved["xgb_best"]
    class_names  = le.classes_

    print(f"  Model   : {model_path}")
    print(f"  Kelas   : {list(class_names)}")

    # Ekstraksi fitur
    tmp_csv = os.path.join(split_plot_dir, "_tmp_features.csv")

    if has_labels:
        df_feat = run_feature_extraction(
            data_dir       = pred_input,
            output_csv     = tmp_csv,
            station_coords = station_coords,
            valid_labels   = list(class_names),
        )
        y_true_str = df_feat["Label"].values if not df_feat.empty else None
    else:
        df_feat    = run_feature_extraction_predict(
            data_dir       = pred_input,
            output_csv     = tmp_csv,
            station_coords = station_coords,
            already_cut    = already_cut,
        )
        y_true_str = None

    if df_feat.empty:
        print(f"  [SKIP] Ekstraksi fitur gagal untuk split ini.")
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)
        return None

    X_new = df_feat.reindex(columns=feature_cols, fill_value=0.0)

    # Prediksi
    svm_pred = svm_best.predict(X_new)
    svm_prob = svm_best.predict_proba(X_new)
    xgb_pred = xgb_best.predict(X_new)
    xgb_prob = xgb_best.predict_proba(X_new)

    svm_label = le.inverse_transform(svm_pred)
    xgb_label = le.inverse_transform(xgb_pred)
    svm_conf  = svm_prob.max(axis=1)
    xgb_conf  = xgb_prob.max(axis=1)

    # Susun output DataFrame
    out_df = df_feat[["Event"]].copy() if "Event" in df_feat.columns \
             else pd.DataFrame({"Event": [f"row_{i}" for i in range(len(X_new))]})
    if has_labels:
        out_df["True_Label"]     = y_true_str
    out_df["SVM_Pred"]       = svm_label
    out_df["SVM_Confidence"] = svm_conf.round(4)
    out_df["XGB_Pred"]       = xgb_label
    out_df["XGB_Confidence"] = xgb_conf.round(4)
    out_df["Agreement"]      = (svm_label == xgb_label)

    csv_out = os.path.join(split_plot_dir, "blind_new_predictions.csv")
    out_df.to_csv(csv_out, index=False)
    print(f"  Prediksi disimpan : {csv_out}")

    # Distribusi prediksi
    _plot_prediction_distribution(
        svm_label, xgb_label, class_names,
        save_path=os.path.join(split_plot_dir, "blind_new_distribution.png"),
    )

    # Evaluasi jika berlabel
    if has_labels and y_true_str is not None:
        y_true_enc = le.transform(y_true_str)

        for name, y_pred in [("SVM", svm_pred), ("XGBoost", xgb_pred)]:
            acc = accuracy_score(y_true_enc, y_pred)
            f1  = f1_score(y_true_enc, y_pred, average="macro", zero_division=0)
            print(f"  {name}: Accuracy={acc:.4f}  F1-macro={f1:.4f}")

        plot_blind_test_results(
            y_true_enc, svm_pred, xgb_pred,
            svm_prob, xgb_prob, class_names,
            output_dir=split_plot_dir,
        )
        # Rename agar nama file konsisten dengan blind_new_*
        for old, new in [
            ("blind_cm_SVM.png",        "blind_new_cm_SVM.png"),
            ("blind_cm_XGBoost.png",    "blind_new_cm_XGBoost.png"),
            ("blind_roc_SVM.png",       "blind_new_roc_SVM.png"),
            ("blind_roc_XGBoost.png",   "blind_new_roc_XGBoost.png"),
            ("blind_f1_comparison.png", "blind_new_f1_comparison.png"),
        ]:
            p_old = os.path.join(split_plot_dir, old)
            p_new = os.path.join(split_plot_dir, new)
            if os.path.exists(p_old):
                os.rename(p_old, p_new)

        _save_blind_new_report(
            y_true_enc, svm_pred, xgb_pred,
            class_names, split_plot_dir,
        )

    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)

    return out_df


# ── Helper: bar chart perbandingan semua split ────────────────────────────────
def _plot_blind_all_comparison(df_summary, has_labels, output_dir):
    """Bar chart SVM Accuracy vs XGBoost Accuracy per split."""
    if "SVM_Accuracy" not in df_summary.columns:
        # Unlabeled: plot agreement & confidence saja
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        splits = df_summary["Split"].tolist()
        x      = np.arange(len(splits))

        axes[0].bar(x, df_summary["Agreement_%"], color="#4472C4", zorder=3)
        axes[0].set_xticks(x); axes[0].set_xticklabels(splits, rotation=20, ha="right")
        axes[0].set_ylabel("Agreement (%)"); axes[0].set_ylim(0, 105)
        axes[0].set_title("Model Agreement per Split"); axes[0].grid(True, alpha=0.3, axis="y")

        width = 0.35
        axes[1].bar(x - width/2, df_summary["SVM_Conf_Mean"],  width, label="SVM",     color="#4472C4")
        axes[1].bar(x + width/2, df_summary["XGB_Conf_Mean"],  width, label="XGBoost", color="#ED7D31")
        axes[1].set_xticks(x); axes[1].set_xticklabels(splits, rotation=20, ha="right")
        axes[1].set_ylabel("Mean Confidence"); axes[1].set_ylim(0, 1.1)
        axes[1].set_title("Mean Confidence per Split"); axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.suptitle("Blind Test Data Baru — Perbandingan Semua Split",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(output_dir, "blind_all_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot: {path}")
        return

    # Labeled: accuracy + confidence
    splits = df_summary["Split"].tolist()
    x      = np.arange(len(splits))
    width  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Accuracy
    b1 = axes[0].bar(x - width/2, df_summary["SVM_Accuracy"],
                     width, label="SVM", color="#4472C4", zorder=3)
    b2 = axes[0].bar(x + width/2, df_summary["XGB_Accuracy"],
                     width, label="XGBoost", color="#ED7D31", zorder=3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].annotate(f"{h:.3f}",
                         xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Accuracy"); axes[0].set_ylim(0, 1.15)
    axes[0].set_title("Blind Test Accuracy per Split")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")

    # Panel 2: Mean Confidence
    axes[1].bar(x - width/2, df_summary["SVM_Conf_Mean"],
                width, label="SVM", color="#4472C4", zorder=3)
    axes[1].bar(x + width/2, df_summary["XGB_Conf_Mean"],
                width, label="XGBoost", color="#ED7D31", zorder=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Mean Confidence"); axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Mean Prediction Confidence per Split")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Blind Test Data Baru — Perbandingan Semua Split",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "blind_all_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {path}")

# ===========================================================================
# MULTI-SPLIT LOOP — Melatih & mengevaluasi 4 skenario split sekaligus
# ===========================================================================

SPLIT_SCENARIOS = [0.4, 0.3, 0.2, 0.1]   # test_size: 60/40, 70/30, 80/20, 90/10

SPLIT_SCENARIOS = [0.4, 0.3, 0.2, 0.1]


def run_all_splits(
    output_csv,
    base_model_output,
    base_output_plots,
    random_state=42,
    splits=None,
    blind_size=0.0,
    blind_seed=99,
    data_dir=None,
    use_fk=True,
):
    """
    Menjalankan SEMUA skenario ML:

      A. Dengan fitur spasial array = 14 fitur
         11 temporal/spektral + Back_Azimuth + Slowness + Beam_Power

      B. Tanpa fitur spasial array = 11 fitur
         11 temporal/spektral saja

    Untuk masing-masing skenario dijalankan:
      60:40, 70:30, 80:20, 90:10
      -> training SVM-RBF
      -> training XGBoost
      -> evaluasi
      -> penyimpanan model, plot, report
    """
    if splits is None:
        splits = SPLIT_SCENARIOS

    # Jika fungsi dipanggil langsung dengan use_fk=False, tetap jalankan
    # hanya skenario 11 fitur untuk kompatibilitas.
    scenarios = [True, False] if use_fk else [False]

    summary_rows = []

    for scenario_fk in scenarios:

        scenario_name = "WITH_ARRAY_14_FEATURES" if scenario_fk else "NO_SPATIAL_11_FEATURES"
        scenario_suffix = "" if scenario_fk else "_no_fk"

        scenario_model_root = base_model_output
        scenario_plot_root = base_output_plots

        print()
        print("╔" + "═" * 60 + "╗")
        print(
            "║  SKENARIO: "
            + (
                "14 FITUR — TEMPORAL/SPEKTRAL + SPASIAL ARRAY"
                if scenario_fk
                else
                "11 FITUR — TEMPORAL/SPEKTRAL TANPA FITUR SPASIAL"
            ).ljust(60)
            + "║"
        )
        print("╚" + "═" * 60 + "╝")

        for test_size in splits:
            train_pct = int(round((1 - test_size) * 100))
            test_pct = int(round(test_size * 100))

            tag = f"split_{train_pct}_{test_pct}{scenario_suffix}"

            model_dir = os.path.join(scenario_model_root, tag)
            plots_dir = os.path.join(scenario_plot_root, tag)
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)

            print()
            print("╔" + "═" * 52 + "╗")
            print(
                f"║  {scenario_name} | Split {train_pct}/{test_pct}"
                .ljust(52) + "║"
            )
            print("╚" + "═" * 52 + "╝")

            # Training
            run_training(
                output_csv=output_csv,
                model_output=model_dir,
                test_size=test_size,
                random_state=random_state,
                blind_size=blind_size,
                blind_seed=blind_seed,
                data_dir=data_dir,
                output_plots=plots_dir,
                tag=tag,
                use_fk=scenario_fk,
            )

            # Evaluation
            run_evaluation(
                output_csv=output_csv,
                model_output=model_dir,
                output_plots=plots_dir,
                test_size=test_size,
                random_state=random_state,
                data_dir=data_dir,
                tag=tag,
            )

            # Ringkasan metrik
            model_path = os.path.join(model_dir, "seismic_models.joblib")
            saved = joblib.load(model_path)

            for model_key, model_name in [
                ("svm_best", "SVM-RBF"),
                ("xgb_best", "XGBoost"),
            ]:
                mdl = saved[model_key]
                y_pred = mdl.predict(saved["X_test"])
                acc = accuracy_score(saved["y_test"], y_pred)
                f1 = f1_score(
                    saved["y_test"],
                    y_pred,
                    average="macro",
                    zero_division=0,
                )

                summary_rows.append({
                    "Feature_Scenario": (
                        "14_features_with_spatial"
                        if scenario_fk
                        else "11_features_without_spatial"
                    ),
                    "Split": f"{train_pct}/{test_pct}",
                    "Model": model_name,
                    "Accuracy": round(acc, 4),
                    "F1_macro": round(f1, 4),
                })

            print(f"✅ Selesai: {scenario_name} — {train_pct}/{test_pct}")
            print(f"   Model → {model_dir}")
            print(f"   Plots → {plots_dir}")

    # ==============================================================
    # SATU TABEL RINGKASAN UNTUK SEMUA SKENARIO
    # ==============================================================
    df_summary = pd.DataFrame(summary_rows)

    os.makedirs(base_output_plots, exist_ok=True)
    summary_path = os.path.join(
        base_output_plots,
        "summary_all_splits_11_vs_14_features.csv",
    )
    df_summary.to_csv(summary_path, index=False)

    print()
    print("╔" + "═" * 60 + "╗")
    print("║  RINGKASAN 11 FITUR vs 14 FITUR — SEMUA SPLIT".ljust(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print(df_summary.to_string(index=False))
    print(f"\n✅ Ringkasan disimpan: {summary_path}")

    return df_summary
# ===========================================================================
# ARGUMENT PARSER
# ===========================================================================

def load_station_coords(station_coords_arg):
    """
    Memuat koordinat stasiun dari argumen --station_coords.

    Mendukung dua format:
      1. JSON string : '{"R0279": [-7.5401, 110.4460], "R265F": [-7.5412, 110.4475]}'
      2. Path file   : ./my_station_coords.json

    Jika argumen None, mengembalikan koordinat default (Merapi Array).

    Parameter
    ---------
    station_coords_arg : str | None

    Return
    ------
    dict  — {kode_stasiun: (lat, lon)}
    """
    if station_coords_arg is None:
        print("  Menggunakan koordinat default (Merapi Array).")
        return DEFAULT_STATION_COORDS

    if os.path.isfile(station_coords_arg):
        with open(station_coords_arg, "r") as f:
            raw = json.load(f)
    else:
        raw = json.loads(station_coords_arg)

    coords = {k: tuple(v) for k, v in raw.items()}
    print(f"  Koordinat dimuat: {list(coords.keys())}")
    return coords


# ===========================================================================
# TAHAP 4 — PREDICTION
# ===========================================================================

def plot_prediction_overview(results_df, class_names, output_dir, tag):
    """
    Plot ringkasan distribusi prediksi:
    - Bar chart distribusi kelas (SVM & XGBoost side-by-side)
    - Pie chart proporsi kelas (XGBoost)
    - Agreement matrix SVM vs XGBoost

    Output: prediction_overview_{tag}.png
    """
    n_classes  = len(class_names)
    SVM_COLOR  = "#4472C4"
    XGB_COLOR  = "#ED7D31"

    svm_counts = results_df["SVM_Pred_Label"].value_counts().reindex(class_names, fill_value=0)
    xgb_counts = results_df["XGB_Pred_Label"].value_counts().reindex(class_names, fill_value=0)
    x     = np.arange(n_classes)
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Ringkasan Prediksi — {tag}", fontsize=13, fontweight="bold")

    # ── Subplot 1: Bar distribusi ──
    ax = axes[0]
    b1 = ax.bar(x - width/2, svm_counts.values, width, label="SVM",
                color=SVM_COLOR, zorder=3)
    b2 = ax.bar(x + width/2, xgb_counts.values, width, label="XGBoost",
                color=XGB_COLOR, zorder=3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(str(int(h)),
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Jumlah Sampel Terprediksi")
    ax.set_title("Distribusi Kelas Prediksi", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Subplot 2: Pie chart (XGBoost) ──
    ax = axes[1]
    colors_pie  = plt.cm.Set2(np.linspace(0, 0.85, n_classes))
    nonzero_idx = xgb_counts.values > 0
    ax.pie(
        xgb_counts.values[nonzero_idx],
        labels=np.array(class_names)[nonzero_idx],
        autopct="%1.1f%%",
        colors=colors_pie[nonzero_idx],
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title("Proporsi Prediksi (XGBoost)", fontsize=11, fontweight="bold")

    # ── Subplot 3: Agreement matrix SVM vs XGBoost ──
    ax = axes[2]
    cls_to_idx = {cn: i for i, cn in enumerate(class_names)}
    agree_mat  = np.zeros((n_classes, n_classes), dtype=int)
    for _, row in results_df.iterrows():
        i = cls_to_idx.get(row["SVM_Pred_Label"], -1)
        j = cls_to_idx.get(row["XGB_Pred_Label"], -1)
        if i >= 0 and j >= 0:
            agree_mat[i, j] += 1
    im = ax.imshow(agree_mat, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("XGBoost Prediksi", fontsize=9)
    ax.set_ylabel("SVM Prediksi", fontsize=9)
    ax.set_title("Agreement Matrix (SVM vs XGBoost)", fontsize=11, fontweight="bold")
    mx = agree_mat.max() if agree_mat.max() > 0 else 1
    for i in range(n_classes):
        for j in range(n_classes):
            v = agree_mat[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=9, color="white" if v > mx * 0.6 else "black")

    plt.tight_layout()
    path = os.path.join(output_dir, f"prediction_overview_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot: {path}")


def plot_prediction_confidence(results_df, class_names, output_dir, tag):
    """
    Plot distribusi confidence (max probability) prediksi:
    - Violin confidence per kelas (SVM & XGBoost)
    - Histogram confidence gabungan

    Output: prediction_confidence_{tag}.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Distribusi Confidence Prediksi — {tag}", fontsize=13, fontweight="bold")

    for ax, (model_name, conf_col, pred_col, color) in zip(
        axes[:2],
        [
            ("SVM",     "SVM_Confidence", "SVM_Pred_Label", "#4472C4"),
            ("XGBoost", "XGB_Confidence", "XGB_Pred_Label", "#ED7D31"),
        ],
    ):
        data_per_class = []
        labels_present = []
        for cn in class_names:
            subset = results_df[results_df[pred_col] == cn][conf_col]
            if len(subset) > 0:
                data_per_class.append(subset.values)
                labels_present.append(cn)
        if data_per_class:
            vp = ax.violinplot(
                data_per_class,
                positions=range(len(data_per_class)),
                showmeans=True, showmedians=True,
            )
            for pc in vp["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            vp["cmeans"].set_color("red")
            vp["cmedians"].set_color("blue")
        ax.set_xticks(range(len(labels_present)))
        ax.set_xticklabels(labels_present, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Confidence (Max Prob.)")
        ax.set_title(f"{model_name} — Confidence per Kelas", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

    # ── Subplot 3: Histogram gabungan ──
    ax = axes[2]
    ax.hist(results_df["SVM_Confidence"], bins=20, alpha=0.6,
            color="#4472C4", label="SVM", edgecolor="white", linewidth=0.5)
    ax.hist(results_df["XGB_Confidence"], bins=20, alpha=0.6,
            color="#ED7D31", label="XGBoost", edgecolor="white", linewidth=0.5)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Threshold 0.5")
    ax.axvline(x=0.8, color="green", linestyle="--", alpha=0.7, label="High conf. 0.8")
    ax.set_xlabel("Confidence (Max Probability)")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Histogram Confidence Keseluruhan", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, f"prediction_confidence_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot: {path}")


def plot_prediction_per_sample(results_df, class_names, model_name,
                                output_dir, tag, max_samples=40):
    """
    Stacked horizontal bar chart probabilitas prediksi per sampel.

    Perbaikan:
    - Deteksi prefix kolom probabilitas secara dinamis
      (mendukung 'SVM_Prob_', 'XGB_Prob_', 'XGBoost_Prob_', dll.)
    - Fallback ke confidence bar jika kolom prob tidak tersedia
    """
    # ── Deteksi prefix kolom probabilitas ────────────────────────────
    def _find_prob_col(df, model_name, class_name):
        candidates = [
            f"{model_name}_Prob_{class_name}",
            f"{model_name.lower()}_Prob_{class_name}",
            f"{model_name.lower()}_prob_{class_name}",
        ]
        # Tambahan: coba tebak prefix dari kolom yang ada
        for col in df.columns:
            col_low = col.lower()
            if "prob" in col_low and class_name.lower() in col_low:
                mn_variants = [
                    model_name.lower(),
                    model_name[:3].lower(),   # misal 'xgb' dari 'XGBoost'
                    model_name.replace("Boost","").lower(),
                ]
                for mv in mn_variants:
                    if col_low.startswith(mv):
                        candidates.append(col)
                        break
        for c in candidates:
            if c in df.columns:
                return c
        return None

    pred_col = f"{model_name}_Pred_Label"
    conf_col = f"{model_name}_Confidence"

    if pred_col not in results_df.columns:
        print(f"   [WARN] Kolom {pred_col} tidak ditemukan, skip per-sample plot.")
        return

    df_plot = results_df.copy()
    if len(df_plot) > max_samples:
        df_plot = df_plot.sample(max_samples, random_state=42).reset_index(drop=True)

    # Cek apakah kolom probabilitas tersedia
    prob_col_map = {
        cn: _find_prob_col(df_plot, model_name, cn)
        for cn in class_names
    }
    has_prob = any(v is not None for v in prob_col_map.values())

    colors = plt.cm.Set2(np.linspace(0, 0.85, len(class_names)))
    n      = len(df_plot)

    fig, ax = plt.subplots(figsize=(13, max(6, n * 0.38)))

    if has_prob:
        # ── Mode 1: Stacked bar probabilitas ─────────────────────────
        left = np.zeros(n)
        for i, cn in enumerate(class_names):
            col = prob_col_map.get(cn)
            if col is None:
                continue
            vals = df_plot[col].fillna(0).values
            bars = ax.barh(range(n), vals, left=left,
                           color=colors[i], label=cn,
                           edgecolor="white", linewidth=0.3)
            # Anotasi nilai jika cukup lebar
            for bar, v in zip(bars, vals):
                if v > 0.12:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.2f}",
                        ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold",
                    )
            left += vals

        ax.set_xlabel("Probabilitas", fontsize=10)
        ax.set_xlim(0, 1.5)

    else:
        # ── Mode 2: Fallback — confidence bar ─────────────────────────
        # Warna bar sesuai kelas prediksi
        CLASS_COLORS = {
            "NonEvent"  : "#2196F3",
            "Multiphase": "#E91E63",
            "Rockfall"  : "#FF9800",
            "VTB"       : "#4CAF50",
        }
        bar_colors = [
            CLASS_COLORS.get(df_plot[pred_col].iloc[i], "#607D8B")
            for i in range(n)
        ]
        conf_vals = df_plot[conf_col].fillna(0).values if conf_col in df_plot.columns \
                    else np.zeros(n)

        ax.barh(range(n), conf_vals, color=bar_colors,
                edgecolor="white", linewidth=0.3)
        ax.axvline(0.8, color="green",  linestyle="--",
                   linewidth=1, alpha=0.7, label="High conf 0.8")
        ax.axvline(0.5, color="orange", linestyle="--",
                   linewidth=1, alpha=0.7, label="Low conf 0.5")
        ax.set_xlabel("Confidence (Max Probability)", fontsize=10)
        ax.set_xlim(0, 1.15)

        # Legend patch per kelas
        patches = [
            mpatches.Patch(color=v, label=k)
            for k, v in CLASS_COLORS.items()
            if k in df_plot[pred_col].values
        ]
        ax.legend(handles=patches, loc="lower right",
                  fontsize=9, title="Kelas Prediksi")

        fig.text(0.5, 0.01,
                 "⚠ Kolom probabilitas tidak tersedia — menampilkan confidence",
                 ha="center", fontsize=8.5, color="gray", style="italic")

    # ── Anotasi label prediksi + confidence di sebelah kanan bar ─────
    if pred_col in df_plot.columns:
        for idx in range(n):
            lbl  = df_plot[pred_col].iloc[idx]
            conf = df_plot[conf_col].iloc[idx] \
                   if conf_col in df_plot.columns else 0.0
            ax.text(
                1.02, idx,
                f"{lbl} ({conf:.2f})",
                va="center", fontsize=7.5, color="black",
            )

    # Sumbu Y: event label
    event_labels = (
        df_plot["Event"].tolist()
        if "Event" in df_plot.columns
        else [str(i) for i in range(n)]
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [str(e)[:35] for e in event_labels], fontsize=7,
    )

    mode_str = "Probabilitas" if has_prob else "Confidence (Fallback)"
    ax.set_title(
        f"{model_name} — {mode_str} per Sampel [{tag}]",
        fontsize=11, fontweight="bold",
    )
    if has_prob:
        ax.legend(loc="lower right", fontsize=9,
                  bbox_to_anchor=(1.45, 0))
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(
        output_dir, f"prediction_per_sample_{model_name}_{tag}.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot: {path}")

def plot_prediction_accuracy_vs_true(results_df, class_names, output_dir, tag):
    """
    Confusion matrix prediksi vs label aktual.
    Hanya dipanggil jika kolom 'True_Label' tersedia.

    Output: prediction_accuracy_{tag}.png
    """
    if "True_Label" not in results_df.columns:
        return

    y_true = results_df["True_Label_Enc"].values
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Evaluasi Prediksi vs Label Aktual — {tag}",
                 fontsize=13, fontweight="bold")

    for ax, (name, pred_col) in zip(axes, [
        ("SVM",     "SVM_Pred_Enc"),
        ("XGBoost", "XGB_Pred_Enc"),
    ]):
        y_pred = results_df[pred_col].values
        cm     = confusion_matrix(y_true, y_pred)
        disp   = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(cmap="Blues", xticks_rotation=30, ax=ax, colorbar=False)
        acc = accuracy_score(y_true, y_pred)
        ax.set_title(f"{name} — Accuracy: {acc:.3f}", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, f"prediction_accuracy_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot: {path}")


# ⚠️ REKONSTRUKSI ⚠️
# Isi asli fungsi ini tidak sempat terbaca sebelum SSD terputus (hanya
# signature & docstring yang sempat terlihat). Implementasi di bawah
# ditulis ulang agar fungsional setara — signature, tempat pemanggilan
# (dari run_prediction), dan tujuan (contoh waveform per kelas dari folder
# prediksi) sama persis — tapi detail kode BUKAN salinan asli Anda.
def plot_prediction_waveform_examples(
    results_df, data_dir, class_names, output_dir, tag,
    n_samples=5, random_state=42,
    paz=None, prefilter=None, target_fs=None, fmin=None, fmax=None,
):
    """
    Plot contoh waveform dari event yang diprediksi, diambil acak per kelas
    hasil prediksi (kolom XGB_Pred_Label), untuk pengecekan visual cepat.

    Preprocessing ringan (detrend + taper + bandpass) TANPA instrument
    correction — cukup untuk tampilan visual, bukan untuk fitur numerik
    (instrument correction penuh dipakai di save_all_waveforms() / pipeline
    ekstraksi fitur yang sebenarnya).

    Segmen yang ditampilkan diambil dari TENGAH sinyal (maks 120 detik)
    supaya tidak kena efek taper di ujung rekaman.

    Output: prediction_waveform_examples_{tag}.png
    """
    if paz       is None: paz       = _DEFAULT_PAZ
    if prefilter is None: prefilter = _PREFILTER
    if target_fs is None: target_fs = _TARGET_FS
    if fmin      is None: fmin      = _FMIN
    if fmax      is None: fmax      = _FMAX

    PLOT_DURATION_S = 120
    rng = np.random.RandomState(random_state)

    class_colors = {
        "NonEvent":   "#2196F3",
        "Multiphase": "#E91E63",
        "Rockfall":   "#FF9800",
        "VTB":        "#4CAF50",
    }

    pred_col = "XGB_Pred_Label" if "XGB_Pred_Label" in results_df.columns \
               else "SVM_Pred_Label"

    # Kumpulkan semua file .mseed di data_dir untuk lookup by event_id
    all_files = glob.glob(os.path.join(data_dir, "**", "*.mseed"), recursive=True)
    if not all_files:
        all_files = glob.glob(os.path.join(data_dir, "*.mseed"))

    def _find_files_for_event(event_id):
        exact = [f for f in all_files if os.path.basename(f).startswith(str(event_id))]
        if exact:
            return exact
        return [f for f in all_files if str(event_id) in os.path.basename(f)]

    samples = []
    for cls in class_names:
        events = results_df.loc[results_df[pred_col] == cls, "Event"].unique().tolist()
        if not events:
            continue
        rng.shuffle(events)
        for ev in events[:max(1, n_samples // max(1, len(class_names)))]:
            files = _find_files_for_event(ev)
            if files:
                samples.append((cls, ev, files))

    if not samples:
        print("   [WARN] Tidak ada sampel waveform yang cocok untuk diplot.")
        return

    n = len(samples)
    fig = plt.figure(figsize=(15, 3.8 * n))
    gs  = gridspec.GridSpec(n, 2, width_ratios=[3, 1], hspace=0.6, wspace=0.3)

    for idx, (cls, event_id, files) in enumerate(samples):
        color = class_colors.get(cls, "#607D8B")
        try:
            st = Stream()
            for f in files:
                st += read(f)
            st.merge(fill_value=0)
            tr = st[0]
            tr.detrend("demean")
            tr.detrend("linear")
            tr.taper(max_percentage=0.05, type="cosine")
            if tr.stats.sampling_rate != target_fs:
                tr.resample(target_fs)
            nyq = tr.stats.sampling_rate / 2.0
            tr.filter("bandpass", freqmin=fmin, freqmax=min(fmax, nyq * 0.9),
                       corners=4, zerophase=True)
            data = tr.data.astype(np.float64)
            sr   = tr.stats.sampling_rate
        except Exception as e:
            print(f"   [WARN] Gagal memuat waveform {event_id} ({cls}): {e}")
            continue

        plot_samples = int(PLOT_DURATION_S * sr)
        if len(data) >= plot_samples:
            start = (len(data) - plot_samples) // 2
            data  = data[start: start + plot_samples]
        t = np.arange(len(data)) / sr

        ax_wave = fig.add_subplot(gs[idx, 0])
        ax_wave.plot(t, data, color=color, linewidth=0.7, alpha=0.9)
        # pad=20 — lihat catatan di _plot_sample_waveforms_for_subset()
        # soal tumpang-tindih judul loc="left" dengan notasi skala sumbu-Y.
        ax_wave.set_title(f"{cls} — Event: {event_id}", fontsize=10,
                          fontweight="bold", loc="left", pad=20)
        ax_wave.set_xlabel("Waktu (detik)", fontsize=8)
        ax_wave.set_ylabel("Amplitudo", fontsize=8)
        ax_wave.tick_params(labelsize=7)
        ax_wave.grid(True, alpha=0.3)
        ax_wave.margins(x=0)

        ax_spec = fig.add_subplot(gs[idx, 1])
        fft_mag = np.abs(np.fft.rfft(data))
        freqs   = np.fft.rfftfreq(len(data), d=1.0 / sr)
        mask    = freqs <= (fmax + 5)
        ax_spec.plot(freqs[mask], fft_mag[mask], color=color, linewidth=0.8, alpha=0.9)
        ax_spec.set_xlabel("Frekuensi (Hz)", fontsize=8)
        ax_spec.set_ylabel("|FFT|", fontsize=8)
        ax_spec.set_title("Spektrum", fontsize=9, pad=4)
        ax_spec.tick_params(labelsize=7)
        ax_spec.grid(True, alpha=0.3)
        ax_spec.margins(x=0)

    fig.suptitle(f"Contoh Waveform Hasil Prediksi — {tag}",
                 fontsize=13, fontweight="bold", y=1.01)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"prediction_waveform_examples_{tag}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot: {path}")


# ⚠️ REKONSTRUKSI ⚠️
# Sama seperti di atas — isi asli tidak sempat terbaca. Ditulis ulang agar
# fungsional setara (menyimpan SETIAP waveform hasil prediksi ke subfolder
# per kelas, dengan preprocessing PENUH/instrument-corrected — konsisten
# dengan preprocess_stream_per_event yang dipakai pipeline fitur).
def save_all_waveforms(
    results_df, data_dir, class_names, output_dir, tag,
    paz=None, prefilter=None, target_fs=None, fmin=None, fmax=None,
):
    """
    Menyimpan waveform INDIVIDUAL untuk setiap event di results_df ke
    subfolder per kelas prediksi:
        output_dir/waveforms_{tag}/{Kelas}/{event_id}.png

    Preprocessing memakai preprocess_stream_per_event() — SAMA PERSIS
    dengan pipeline ekstraksi fitur — supaya waveform yang tersimpan
    mencerminkan sinyal yang benar-benar dipakai model.
    """
    if paz       is None: paz       = _DEFAULT_PAZ
    if prefilter is None: prefilter = _PREFILTER
    if target_fs is None: target_fs = _TARGET_FS
    if fmin      is None: fmin      = _FMIN
    if fmax      is None: fmax      = _FMAX

    pred_col = "XGB_Pred_Label" if "XGB_Pred_Label" in results_df.columns \
               else "SVM_Pred_Label"
    conf_col = "XGB_Confidence" if "XGB_Confidence" in results_df.columns \
               else "SVM_Confidence"

    all_files = glob.glob(os.path.join(data_dir, "**", "*.mseed"), recursive=True)
    if not all_files:
        all_files = glob.glob(os.path.join(data_dir, "*.mseed"))

    def _find_files_for_event(event_id):
        exact = [f for f in all_files if os.path.basename(f).startswith(str(event_id))]
        if exact:
            return exact
        return [f for f in all_files if str(event_id) in os.path.basename(f)]

    out_root = os.path.join(output_dir, f"waveforms_{tag}")
    os.makedirs(out_root, exist_ok=True)

    saved_count, failed_count = 0, 0
    for _, row in results_df.iterrows():
        event_id = row.get("Event", None)
        cls      = row.get(pred_col, "Unknown")
        conf     = row.get(conf_col, 0.0)
        if event_id is None:
            continue

        files = _find_files_for_event(event_id)
        if not files:
            failed_count += 1
            continue

        try:
            st_raw = Stream()
            for f in files:
                st_raw += read(f)
            st = preprocess_stream_per_event(
                st_raw, paz=paz, prefilter=prefilter,
                target_fs=target_fs, fmin=fmin, fmax=fmax,
            )
            if len(st) == 0:
                raise ValueError("preprocessing menghasilkan stream kosong")
            tr   = st[0]
            data = tr.data.astype(np.float64)
            sr   = tr.stats.sampling_rate
        except Exception as e:
            print(f"   [WARN] Gagal proses waveform {event_id}: {e}")
            failed_count += 1
            continue

        cls_dir = os.path.join(out_root, str(cls))
        os.makedirs(cls_dir, exist_ok=True)

        t = np.arange(len(data)) / sr
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(t, data, color="#4472C4", linewidth=0.7)
        # pad ekstra — jaga-jaga terhadap notasi skala sumbu-Y (mis. "1e-9")
        # yang muncul di pojok kiri-atas untuk data amplitudo ber-orde kecil.
        ax.set_title(f"{event_id}  —  {cls} (conf={conf:.2f})",
                     fontsize=10, fontweight="bold", pad=14)
        ax.set_xlabel("Waktu (detik)", fontsize=8)
        ax.set_ylabel("Amplitudo (m/s)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
        plt.tight_layout()

        safe_name = "".join(c for c in str(event_id) if c.isalnum() or c in "_-.")
        fig_path = os.path.join(cls_dir, f"{safe_name}.png")
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        saved_count += 1

    print(f"   Waveform disimpan: {saved_count} berhasil, {failed_count} gagal → {out_root}")


# ⚠️ REKONSTRUKSI ⚠️
# Isi asli tidak sempat terbaca. Pelengkap plot_prediction_overview() —
# menampilkan distribusi confidence per kelas dalam bentuk box plot
# berdampingan SVM vs XGBoost, konsisten gaya dengan fungsi lain di file ini.
def plot_prediction_distribution(results_df, class_names, output_dir, tag):
    """
    Box plot probabilitas prediksi (SVM_Prob_/XGB_Prob_) per kelas — melihat
    seberapa jelas model memisahkan tiap kelas.

    Output: prediction_distribution_{tag}.png
    """
    fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 5),
                             sharey=True)
    if len(class_names) == 1:
        axes = [axes]
    fig.suptitle(f"Distribusi Probabilitas per Kelas — {tag}",
                 fontsize=13, fontweight="bold")

    for ax, cn in zip(axes, class_names):
        svm_col = f"SVM_Prob_{cn}"
        xgb_col = f"XGB_Prob_{cn}"
        data, labels = [], []
        if svm_col in results_df.columns:
            data.append(results_df[svm_col].values)
            labels.append("SVM")
        if xgb_col in results_df.columns:
            data.append(results_df[xgb_col].values)
            labels.append("XGBoost")
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], ["#4472C4", "#ED7D31"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_title(cn, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Probabilitas", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, f"prediction_distribution_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot: {path}")


# ⚠️ REKONSTRUKSI ⚠️
# Isi asli tidak sempat terbaca. Pelengkap plot_prediction_confidence() —
# scatter SVM_Confidence vs XGB_Confidence, diwarnai status agreement,
# untuk melihat pola perbedaan confidence saat kedua model tidak sepakat.
def plot_prediction_confidence_detail(results_df, class_names, output_dir, tag):
    """
    Scatter plot SVM_Confidence vs XGB_Confidence, dipisah warna
    berdasarkan agreement (sepakat / tidak sepakat).

    Output: prediction_confidence_detail_{tag}.png
    """
    agree = (results_df["SVM_Pred_Label"] == results_df["XGB_Pred_Label"])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        results_df.loc[agree, "SVM_Confidence"],
        results_df.loc[agree, "XGB_Confidence"],
        color="#4CAF50", alpha=0.6, s=30, label=f"Sepakat ({agree.sum()})",
    )
    ax.scatter(
        results_df.loc[~agree, "SVM_Confidence"],
        results_df.loc[~agree, "XGB_Confidence"],
        color="#E91E63", alpha=0.7, s=30, label=f"Tidak sepakat ({(~agree).sum()})",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("SVM Confidence", fontsize=11)
    ax.set_ylabel("XGBoost Confidence", fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Confidence SVM vs XGBoost — {tag}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, f"prediction_confidence_detail_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot: {path}")


# ⚠️ REKONSTRUKSI ⚠️
# Isi asli tidak sempat terbaca. Laporan teks bergaya sama dengan
# save_evaluation_report()/save_blind_test_report() di atas.
def save_prediction_report(results_df, class_names, test_size, output_dir, tag):
    """
    Menyimpan ringkasan hasil prediksi ke file .txt: distribusi kelas,
    confidence rata-rata, agreement, dan (jika ada True_Label) akurasi.

    Output: prediction_report_{tag}.txt
    """
    sep      = "=" * 70
    sep_thin = "-" * 70
    n        = len(results_df)

    lines = [
        sep,
        "  SEISMIC EVENT CLASSIFICATION — PREDICTION REPORT",
        f"  Split model : {tag}",
        f"  Generated   : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Jumlah sampel prediksi : {n}",
        f"  Kelas       : {list(class_names)}",
        sep, "",
    ]

    for name, pred_col, conf_col in [
        ("SVM", "SVM_Pred_Label", "SVM_Confidence"),
        ("XGBoost", "XGB_Pred_Label", "XGB_Confidence"),
    ]:
        lines += [sep_thin, f"  MODEL: {name}", sep_thin, ""]
        dist = results_df[pred_col].value_counts().reindex(class_names, fill_value=0)
        for cn, cnt in dist.items():
            lines.append(f"    {cn:<15}: {cnt:5d} ({100*cnt/max(n,1):.1f}%)")
        lines += [
            f"    Mean confidence   : {results_df[conf_col].mean():.4f}",
            f"    Median confidence : {results_df[conf_col].median():.4f}",
            f"    High conf (>0.8)  : {(results_df[conf_col] > 0.8).mean()*100:.1f}%",
            "",
        ]

    agreement = (results_df["SVM_Pred_Label"] == results_df["XGB_Pred_Label"]).mean()
    lines += [sep_thin, f"  Agreement SVM vs XGBoost: {agreement*100:.1f}%", sep_thin, ""]

    if "True_Label" in results_df.columns:
        y_true = results_df["True_Label_Enc"].values
        for name, pred_col in [("SVM", "SVM_Pred_Enc"), ("XGBoost", "XGB_Pred_Enc")]:
            y_pred = results_df[pred_col].values
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
            lines += [
                sep_thin, f"  EVALUASI vs True_Label — {name}", sep_thin,
                f"  Accuracy : {acc:.4f}  |  F1-macro : {f1:.4f}", "",
            ]
            for ln in classification_report(
                y_true, y_pred, target_names=class_names, digits=3
            ).splitlines():
                lines.append("  " + ln)
            lines.append("")

    lines += [sep, "  END OF PREDICTION REPORT", sep]

    path = os.path.join(output_dir, f"prediction_report_{tag}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"   Report: {path}")


def run_prediction(pred_input, model_output, output_plots,
                   station_coords=None, tag="", random_state=42):

    print()
    print("=" * 60)
    print(f"TAHAP 4: PREDIKSI  [{tag}]")
    print("=" * 60)
    os.makedirs(output_plots, exist_ok=True)

    # ── 1. Load model ─────────────────────────────────────────────────
    model_path = os.path.join(model_output, "seismic_models.joblib")
    if not os.path.exists(model_path):
        print(f"  [ERROR] Model tidak ditemukan: {model_path}")
        print("  Pastikan --mode all_splits sudah dijalankan terlebih dahulu.")
        sys.exit(1)

    saved       = joblib.load(model_path)
    le          = saved["label_encoder"]
    num_cols    = saved["feature_cols"]
    svm_best    = saved["svm_best"]
    xgb_best    = saved["xgb_best"]
    test_size   = saved.get("test_size", TEST_SIZE)
    class_names = le.classes_

    # ── DETEKSI OTOMATIS: apakah model ini pakai FK atau tidak? ───────
    # feature_cols yang tersimpan di model adalah sumber kebenaran —
    # jika tidak mengandung 'Back_Azimuth', berarti model _no_fk (11 fitur)
    FK_COLS  = ["Back_Azimuth", "Slowness", "Beam_Power"]
    use_fk   = any(c in num_cols for c in FK_COLS)
    print(f"  Model         : {model_path}")
    print(f"  Fitur model   : {len(num_cols)} kolom | FK: {'✅ Ya' if use_fk else '❌ Tidak (no_fk)'}")
    print(f"  Kelas         : {list(class_names)}")

    # ── 2. Load / Ekstraksi fitur dari input ──────────────────────────
    tmp_csv = os.path.join(output_plots, "_pred_features_tmp.csv")

    if os.path.isfile(pred_input) and pred_input.lower().endswith(".csv"):
        # Input sudah CSV — langsung pakai
        df_raw = pd.read_csv(pred_input)
        print(f"  Input CSV: {pred_input} ({len(df_raw)} baris)")

    elif os.path.isdir(pred_input):
        if station_coords is None:
            station_coords = DEFAULT_STATION_COORDS

        has_label_subfolders = any(
            os.path.isdir(os.path.join(pred_input, lbl))
            for lbl in VALID_LABELS
        )

        if has_label_subfolders:
            # Ada subfolder label → pakai run_feature_extraction (dengan/tanpa FK)
            if use_fk:
                df_raw = run_feature_extraction(
                    data_dir     = pred_input,
                    output_csv   = tmp_csv,
                    station_coords = station_coords,
                    valid_labels = list(class_names),
                )
            else:
                # Model no_fk → ekstrak tanpa FK, lalu drop kolom FK jika ada
                df_raw = run_feature_extraction(
                    data_dir     = pred_input,
                    output_csv   = tmp_csv,
                    station_coords = station_coords,
                    valid_labels = list(class_names),
                )
                # Drop kolom FK jika ikut terekstrak
                df_raw = df_raw.drop(columns=[c for c in FK_COLS if c in df_raw.columns],
                                     errors="ignore")
        else:
            # Flat folder → pakai run_feature_extraction_predict
            df_raw = run_feature_extraction_predict(
                data_dir     = pred_input,
                output_csv   = tmp_csv,
                station_coords = station_coords,
                already_cut  = True,
            )
            if df_raw is not None and not df_raw.empty:
                if use_fk:
                    # Model pakai FK tapi ekstraksi predict mungkin hasilkan nan
                    # → pastikan kolom FK ada (sudah ada dari run_feature_extraction_predict)
                    pass
                else:
                    # Model no_fk → drop kolom FK
                    df_raw = df_raw.drop(
                        columns=[c for c in FK_COLS if c in df_raw.columns],
                        errors="ignore"
                    )
    else:
        df_raw = pd.read_csv(pred_input)
        print(f"  Input fallback CSV: {pred_input} ({len(df_raw)} baris)")

    # ── Guard: cek data ada ───────────────────────────────────────────
    if df_raw is None or len(df_raw) == 0:
        print("\n  [ERROR] Tidak ada data yang berhasil diekstrak dari input.")
        return None

    # ── 3. Cek kolom fitur — gunakan reindex agar tidak error ─────────
    missing = [c for c in num_cols if c not in df_raw.columns]
    if missing:
        print(f"  [WARN] Kolom berikut tidak ada di input, diisi 0: {missing}")
        # Jangan sys.exit — gunakan reindex dengan fillvalue=0
        # agar pipeline tetap jalan (model akan handle dengan nilai 0)

    X_pred      = df_raw.reindex(columns=num_cols, fill_value=0.0)
    event_names = df_raw["Event"].tolist() if "Event" in df_raw.columns \
                  else [f"Sample_{i}" for i in range(len(df_raw))]
    has_true    = "Label" in df_raw.columns and \
                  df_raw["Label"].isin(le.classes_).all()
    y_true      = le.transform(df_raw["Label"]) if has_true else None

    print(f"  Jumlah sampel : {len(X_pred)}")
    print(f"  Label aktual  : {'Tersedia' if has_true else 'Tidak tersedia'}")
    print()

    # 3. Prediksi
    svm_pred_enc = svm_best.predict(X_pred)
    svm_prob     = svm_best.predict_proba(X_pred)
    xgb_pred_enc = xgb_best.predict(X_pred)
    xgb_prob     = xgb_best.predict_proba(X_pred)

    svm_pred_lbl = le.inverse_transform(svm_pred_enc)
    xgb_pred_lbl = le.inverse_transform(xgb_pred_enc)
    svm_conf     = np.max(svm_prob, axis=1)
    xgb_conf     = np.max(xgb_prob, axis=1)

    # 4. Susun DataFrame hasil
    results = {
        "Event"          : event_names,
        "SVM_Pred_Label" : svm_pred_lbl,
        "SVM_Pred_Enc"   : svm_pred_enc,
        "SVM_Confidence" : np.round(svm_conf, 4),
        "XGB_Pred_Label" : xgb_pred_lbl,
        "XGB_Pred_Enc"   : xgb_pred_enc,
        "XGB_Confidence" : np.round(xgb_conf, 4),
    }
    for i, cn in enumerate(class_names):
        results[f"SVM_Prob_{cn}"] = np.round(svm_prob[:, i], 4)
        results[f"XGB_Prob_{cn}"] = np.round(xgb_prob[:, i], 4)
    if has_true:
        results["True_Label"]     = le.inverse_transform(y_true)
        results["True_Label_Enc"] = y_true

    results_df = pd.DataFrame(results)

    # 5. Simpan CSV
    csv_path = os.path.join(output_plots, f"prediction_results_{tag}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  CSV hasil    : {csv_path}")

    # Ringkasan konsol
    n         = len(results_df)
    agreement = (results_df["SVM_Pred_Label"] == results_df["XGB_Pred_Label"]).sum()
    print(f"\n  Agreement (SVM == XGBoost) : {agreement}/{n} ({agreement/n*100:.1f}%)")
    print(f"  SVM  mean confidence       : {svm_conf.mean():.4f}")
    print(f"  XGB  mean confidence       : {xgb_conf.mean():.4f}")
    print()

    # 6. Plot
    print("  Membuat plot prediksi...")
    plot_prediction_overview(results_df, class_names, output_plots, tag)

    # ── Baru: distribusi detail ──────────────────────────────────────
    plot_prediction_distribution(
        results_df, class_names, output_plots, tag,
    )

    # ── Baru: confidence detail ──────────────────────────────────────
    plot_prediction_confidence_detail(
        results_df, class_names, output_plots, tag,
    )

    # ── Lama: confidence ringkasan ───────────────────────────────────
    plot_prediction_confidence(results_df, class_names, output_plots, tag)

    plot_prediction_per_sample(
        results_df, class_names, "SVM", output_plots, tag,
    )
    plot_prediction_per_sample(
        results_df, class_names, "XGBoost", output_plots, tag,
    )

    if has_true:
        plot_prediction_accuracy_vs_true(
            results_df, class_names, output_plots, tag,
        )

    # ── Baru: waveform per kelas ─────────────────────────────────────
    # Hanya jika pred_input adalah folder .mseed
    # BARU — sesuai signature fungsi versi terbaru
    if os.path.isdir(pred_input):
        print("  Membuat plot waveform contoh (5 random)...")
        plot_prediction_waveform_examples(
            results_df=results_df,
            data_dir=pred_input,
            class_names=class_names,
            output_dir=output_plots,
            tag=tag,
            n_samples=5,
            random_state=42,
        )
        # Simpan semua waveform ke subfolder per kelas
        print("\n Menyimpan semua waveform individu...")
        save_all_waveforms(
            results_df   = results_df,
            data_dir     = pred_input,
            class_names  = class_names,
            output_dir   = output_plots,
            tag          = tag,
            paz          = _DEFAULT_PAZ,
            prefilter    = _PREFILTER,
            target_fs    = _TARGET_FS,
            fmin         = _FMIN,
            fmax         = _FMAX,
        )

    # 7. Report
    print("  Menyimpan prediction report (.txt)...")
    save_prediction_report(results_df, class_names, test_size, output_plots, tag)

    print(f"\n✅ Prediksi selesai — output di: {output_plots}")
    return results_df


# ⚠️ REKONSTRUKSI ⚠️
# Isi asli tidak sempat terbaca sebelum SSD terputus. Ditulis ulang agar
# fungsional setara — signature (df_summary, output_dir) dan tempat
# pemanggilan (dari run_all_predictions, lihat di bawah) sama persis.
def plot_prediction_cross_split(df_summary, output_dir):
    """
    Bar chart perbandingan ringkasan prediksi (agreement rate & mean
    confidence) SVM vs XGBoost di seluruh skenario split model.

    df_summary diharapkan punya kolom: Split, Model, AgreementRate, MeanConf
    (lihat run_all_predictions() untuk struktur lengkapnya).

    Output: prediction_summary_chart.png
    """
    svm_rows = df_summary[df_summary["Model"] == "SVM"].sort_values("Split")
    xgb_rows = df_summary[df_summary["Model"] == "XGBoost"].sort_values("Split")

    splits = svm_rows["Split"].tolist()
    x      = np.arange(len(splits))
    width  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Agreement rate
    axes[0].bar(x - width/2, svm_rows["AgreementRate"].values, width,
                label="SVM", color="#4472C4", zorder=3)
    axes[0].bar(x + width/2, xgb_rows["AgreementRate"].values, width,
                label="XGBoost", color="#ED7D31", zorder=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Agreement Rate")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Agreement Rate per Split", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Panel 2: Mean confidence
    axes[1].bar(x - width/2, svm_rows["MeanConf"].values, width,
                label="SVM", color="#4472C4", zorder=3)
    axes[1].bar(x + width/2, xgb_rows["MeanConf"].values, width,
                label="XGBoost", color="#ED7D31", zorder=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Mean Confidence")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Mean Confidence per Split", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Perbandingan Prediksi — Semua Skenario Split",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "prediction_summary_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Plot: {path}")


def run_all_predictions(
    pred_input,
    base_model_output,
    base_output_plots,
    station_coords=None,
    splits=None,
    random_state=42,
):
    """
    Menjalankan prediksi untuk SEMUA sub-folder model yang ditemukan
    di base_model_output secara otomatis.

    Perubahan dari versi lama
    -------------------------
    Versi lama hardcode suffix '_no_fk' sehingga hanya memproses
    split_XX_YY_no_fk. Versi ini mendeteksi otomatis semua sub-folder
    yang berisi seismic_models.joblib — mendukung:
      - split_60_40, split_60_40_no_fk
      - split_70_30, split_70_30_no_fk
      - split_80_20, split_80_20_no_fk
      - split_90_10, split_90_10_no_fk
      - format lain apapun selama ada seismic_models.joblib di dalamnya

    Parameter
    ---------
    pred_input        : str   — CSV fitur atau folder .mseed data prediksi.
    base_model_output : str   — Folder INDUK yang berisi semua sub-folder model.
    base_output_plots : str   — Folder induk output prediksi.
    station_coords    : dict  — Koordinat stasiun (perlu jika input .mseed).
    splits            : list[str] | None
                        Daftar nama sub-folder spesifik yang ingin diproses.
                        None = deteksi otomatis semua sub-folder.
    random_state      : int   — Seed reproducibility.

    Struktur Output
    ---------------
    base_output_plots/
    └── prediction/
        ├── split_60_40/          ← hasil predict model split_60_40
        ├── split_60_40_no_fk/    ← hasil predict model split_60_40_no_fk
        ├── split_70_30/
        ├── split_70_30_no_fk/
        ├── ...
        ├── prediction_summary_all_splits.csv
        └── prediction_summary_chart.png
    """
    print()
    print("=" * 58)
    print("PREDIKSI — SEMUA SKENARIO MODEL")
    print("=" * 58)
    print(f"  Input prediksi : {pred_input}")
    print(f"  Model base dir : {base_model_output}")
    print()

    # ── Deteksi sub-folder otomatis ───────────────────────────────────
    if splits is None:
        splits = sorted([
            d for d in os.listdir(base_model_output)
            if os.path.isdir(os.path.join(base_model_output, d))
            and os.path.exists(
                os.path.join(base_model_output, d, "seismic_models.joblib")
            )
        ])

    if not splits:
        print(f"  [ERROR] Tidak ada sub-folder model ditemukan di: {base_model_output}")
        print("  Pastikan setiap sub-folder berisi seismic_models.joblib")
        return None

    print(f"  Ditemukan {len(splits)} model: {splits}")
    print()

    pred_base_dir = os.path.join(base_output_plots, "prediction")
    os.makedirs(pred_base_dir, exist_ok=True)

    summary_rows = []
    all_results  = {}

    for split_tag in splits:
        model_dir  = os.path.join(base_model_output, split_tag)
        plots_dir  = os.path.join(pred_base_dir, split_tag)
        model_path = os.path.join(model_dir, "seismic_models.joblib")

        # Sudah diverifikasi saat deteksi, tapi double-check
        if not os.path.exists(model_path):
            print(f"  [SKIP] {split_tag}: seismic_models.joblib tidak ditemukan.")
            continue

        os.makedirs(plots_dir, exist_ok=True)
        print(f"  Skenario: {split_tag}")

        results_df = run_prediction(
            pred_input   = pred_input,
            model_output = model_dir,
            output_plots = plots_dir,
            station_coords = station_coords,
            tag          = split_tag,
            random_state = random_state,
        )

        if results_df is None or len(results_df) == 0:
            print(f"  [SKIP] {split_tag}: tidak ada hasil prediksi.")
            continue

        all_results[split_tag] = results_df

        # ── Kumpulkan metrik ringkasan dengan nama kolom yang aman ─────
        # Deteksi nama kolom prediksi secara fleksibel
        svm_pred_col = next(
            (c for c in results_df.columns
             if "svm" in c.lower() and "pred" in c.lower() and "enc" not in c.lower()
             and "prob" not in c.lower()),
            None
        )
        xgb_pred_col = next(
            (c for c in results_df.columns
             if ("xgb" in c.lower() or "xgboost" in c.lower())
             and "pred" in c.lower() and "enc" not in c.lower()
             and "prob" not in c.lower()),
            None
        )
        svm_conf_col = next(
            (c for c in results_df.columns
             if "svm" in c.lower() and "conf" in c.lower()),
            None
        )
        xgb_conf_col = next(
            (c for c in results_df.columns
             if ("xgb" in c.lower() or "xgboost" in c.lower())
             and "conf" in c.lower()),
            None
        )

        if svm_pred_col is None or xgb_pred_col is None:
            print(f"  [WARN] {split_tag}: kolom prediksi tidak ditemukan "
                  f"di DataFrame ({list(results_df.columns)[:8]}...), skip summary.")
            continue

        n         = len(results_df)
        agreement = (results_df[svm_pred_col] == results_df[xgb_pred_col]).mean()

        # Load class names dari model untuk proporsi per kelas
        model_path = os.path.join(model_dir, "seismic_models.joblib")
        saved_mdl  = joblib.load(model_path)
        class_names_local = saved_mdl["label_encoder"].classes_
        test_size_local   = saved_mdl.get("test_size", TEST_SIZE)

        for model_name, pred_col, conf_col in [
            ("SVM",     svm_pred_col, svm_conf_col),
            ("XGBoost", xgb_pred_col, xgb_conf_col),
        ]:
            row = {
                "Split":        split_tag,
                "Model":        model_name,
                "NSamples":     n,
                "AgreementRate": round(agreement, 4),
                "MeanConf":     round(results_df[conf_col].mean(), 4)
                                if conf_col else 0.0,
                "MedianConf":   round(results_df[conf_col].median(), 4)
                                if conf_col else 0.0,
                "HighConfRate": round((results_df[conf_col] > 0.8).mean(), 4)
                                if conf_col else 0.0,
            }
            dist = results_df[pred_col].value_counts(normalize=True)
            for cn in class_names_local:
                row[f"Pred_{cn}"] = round(dist.get(cn, 0) * 100, 1)
            summary_rows.append(row)

        print(f"  Selesai {split_tag}")

    if not summary_rows:
        print("  [WARN] Tidak ada hasil prediksi. Cek folder model.")
        return None

    # ── Ringkasan CSV + chart ─────────────────────────────────────────
    df_summary   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(pred_base_dir, "prediction_summary_all_splits.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n✅ Summary CSV  : {summary_path}")

    plot_prediction_cross_split(df_summary, pred_base_dir)

    return df_summary

# ===========================================================================
# ARGUMENT PARSER
# ===========================================================================

def parse_args():
    """
    Mendefinisikan dan mem-parsing argumen command-line untuk pipeline seismik.

    Argumen
    -------
    --mode           : (wajib) Tahap pipeline.
                       Pilihan: extract | train | evaluate | all | all_splits |
                                predict | predict_all | blind_test | blind_new |
                                blind_new_all
    --data_dir       : Folder MiniSEED. Wajib untuk mode extract/all.
    --output_csv     : Path CSV fitur (output extract / input train & evaluate).
                       Default: ./output/features.csv
    --model_output   : Folder model .joblib. Default: ./models
    --output_plots   : Folder plot evaluasi.   Default: ./plots
    --test_size      : Proporsi data test [0.0–1.0]. Default: 0.4
    --random_state   : Random seed. Default: 42
    --station_coords : Koordinat stasiun (JSON string atau path .json).
                       Format: '{"KODE": [lat, lon], ...}'
    --labels         : Daftar label kelas valid (nama subfolder).
                       Default: Multiphase Rockfall VTB NonEvent
    --no_fk          : Nonaktifkan fitur FK Analysis/Beamforming.

    Contoh Penggunaan
    -----------------
    # One-click full pipeline:
    python seismic_ml_pipeline.py --mode all \\
        --data_dir /data/mseed \\
        --output_csv /output/features.csv \\
        --model_output /models \\
        --output_plots /plots

    # Hanya ekstraksi fitur:
    python seismic_ml_pipeline.py --mode extract \\
        --data_dir /data/mseed --output_csv /output/features.csv

    # Dengan koordinat stasiun custom:
    python seismic_ml_pipeline.py --mode all --data_dir /data \\
        --output_csv /output/features.csv \\
        --station_coords '{"STAT1": [-7.5, 110.4]}'

    # Tanpa fitur FK (Back_Azimuth/Slowness/Beam_Power):
    python seismic_ml_pipeline.py --mode extract --no_fk \\
        --data_dir /data/mseed --output_csv /output/features.csv
    python seismic_ml_pipeline.py --mode all_splits --no_fk \\
        --output_csv /output/features.csv \\
        --model_output /models --output_plots /plots --data_dir /data/mseed
    """
    parser = argparse.ArgumentParser(
        description="Seismic Event Classification Pipeline (SVM + XGBoost)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=parse_args.__doc__,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["extract", "train", "evaluate", "all", "all_splits", "predict", "predict_all", "blind_test", "blind_new", "blind_new_all"],
        help="Tahap pipeline: extract | train | evaluate | all | all_splits | predict | predict_all | blind_test | blind_new | blind_new_all",
    )
    parser.add_argument(
        "--data_dir", default=None,
        help="Folder MiniSEED (subfolder berlabel). Wajib untuk mode extract/all.",
    )
    parser.add_argument(
        "--output_csv", default="./output/features.csv",
        help="Path file CSV fitur. Default: ./output/features.csv",
    )
    parser.add_argument(
        "--model_output", default="./models",
        help="Folder model .joblib. Default: ./models",
    )
    parser.add_argument(
        "--output_plots", default="./plots",
        help="Folder plot evaluasi. Default: ./plots",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.4,
        help="Proporsi data test (0–1). Default: 0.4",
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--station_coords", default=None,
        help='Koordinat stasiun. JSON string atau path .json. Format: {"KODE": [lat, lon]}',
    )
    parser.add_argument(
        "--labels", nargs="+", default=VALID_LABELS,
        help=f"Nama subfolder label kelas. Default: {VALID_LABELS}",
    )
    parser.add_argument(
        "--pred_input", default=None,
        help=(
            "Input data untuk prediksi. "
            "Bisa berupa path file .csv (fitur) atau folder .mseed. "
            "Wajib untuk mode predict / predict_all."
        ),
    )
    parser.add_argument(
        "--blind_size", type=float, default=0.0,
        help=(
            "Proporsi data yang disisihkan sebagai blind holdout SEBELUM "
            "train/test split (0-1). Default: 0.0 (tidak ada blind holdout, "
            "semua data dipakai train/test — pakai --mode blind_new dengan "
            "dataset terpisah untuk blind test). Isi >0 (mis. 0.15) hanya "
            "jika ingin blind test dari potongan dataset yang SAMA."
        ),
    )
    parser.add_argument(
        "--blind_seed", type=int, default=99,
        help="Random seed blind test (harus BERBEDA dari --random_state). Default: 99",
    )
    parser.add_argument(
        "--has_labels", action="store_true", default=False,
        help="Gunakan flag ini jika data baru punya subfolder label.",
    )
    parser.add_argument(
        "--already_cut", action="store_true", default=True,
        help="True jika data .mseed sudah dipotong per event (default: True).",
    )
    parser.add_argument(
        "--no_fk", action="store_true", default=False,
        help=(
            "Gunakan hanya 11 fitur temporal/spektral tanpa menambahkan "
            "Back_Azimuth, Slowness, dan Beam_Power pada model. Untuk "
            "--mode all_splits, pipeline otomatis menjalankan kedua skenario "
            "(14 fitur dan 11 fitur), sehingga flag ini tidak diperlukan."
        ),
    )
    return parser.parse_args()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    """
    Entry point utama — mengorkestrasi seluruh pipeline sesuai argumen --mode.

    Mode Eksekusi
    -------------
    all      : extract → train → evaluate  (one-click, end-to-end)
    extract  : hanya baca .mseed → ekstrak fitur → simpan CSV
    train    : hanya load CSV → latih SVM & XGBoost → simpan model
    evaluate : hanya load model → evaluasi → simpan semua plot
    blind_test : hanya load data blind → evaluasi → simpan semua plot
    """
    args = parse_args()

    # ── Flag --no_fk: otomatis suffix "_no_fk" pada path CSV fitur ────────
    # Supaya extract & all_splits/train yang dijalankan dengan --no_fk selalu
    # menulis/membaca CSV yang berbeda dari versi dengan-FK (tidak saling
    # menimpa), tanpa harus ganti --output_csv manual setiap kali.
    use_fk = not args.no_fk
    if args.no_fk:
        base, ext = os.path.splitext(args.output_csv)
        args.output_csv = f"{base}_no_fk{ext or '.csv'}"

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   Seismic Event Classification Pipeline 2026    ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Mode       : {args.mode.upper()}")
    print(f"  Fitur FK   : {'Aktif' if use_fk else 'NONAKTIF (no_fk)'}")
    print(f"  Output CSV : {args.output_csv}")
    print(f"  Models dir : {args.model_output}")
    print(f"  Plots dir  : {args.output_plots}")
    print()

    station_coords = load_station_coords(args.station_coords)

    if args.mode in ("extract", "all"):
        if args.data_dir is None:
            print("[ERROR] --data_dir wajib untuk mode extract/all.")
            sys.exit(1)
        run_feature_extraction(
            data_dir=args.data_dir,
            output_csv=args.output_csv,
            station_coords=station_coords,
            valid_labels=args.labels,
            use_fk=use_fk,
        )

    if args.mode in ("train", "all"):
        run_training(
            output_csv=args.output_csv,
            model_output=args.model_output,
            test_size=args.test_size,
            random_state=args.random_state,
            blind_size=args.blind_size,
            blind_seed=args.blind_seed,
            data_dir=args.data_dir,
            output_plots=args.output_plots,
            tag=os.path.basename(os.path.normpath(args.model_output)),
            use_fk=use_fk,
        )

    if args.mode in ("evaluate", "all"):
        run_evaluation(
            output_csv=args.output_csv,
            model_output=args.model_output,
            output_plots=args.output_plots,
            random_state=args.random_state,
            data_dir=args.data_dir,
            tag=os.path.basename(os.path.normpath(args.model_output)),
        )

    if args.mode == "all_splits":
        # ==============================================================
        # ONE-CLICK ALL SPLITS:
        # 1) otomatis ekstrak 14 fitur dari waveform hasil beamforming
        # 2) gunakan CSV 14 fitur yang sama untuk:
        #      - model 14 fitur
        #      - model 11 fitur (3 fitur spasial dibuang saat training)
        # 3) jalankan split 60/40, 70/30, 80/20, 90/10
        # ==============================================================
        if args.data_dir is None:
            print("[ERROR] --data_dir wajib untuk mode all_splits.")
            sys.exit(1)

        if args.no_fk:
            print(
                "[INFO] --no_fk diabaikan pada mode all_splits karena "
                "mode ini otomatis menjalankan KEDUA skenario: 14 dan 11 fitur."
            )

        if not os.path.exists(args.data_dir):
            print(f"[ERROR] Folder data tidak ditemukan: {args.data_dir}")
            sys.exit(1)

        print()
        print("=" * 64)
        print("ONE-CLICK ALL SPLITS — EKSTRAKSI + ML 11 FITUR & 14 FITUR")
        print("=" * 64)
        print("  Tahap 1 : Beamforming → 1 waveform representatif → 14 fitur")
        print("  Tahap 2 : ML skenario 14 fitur")
        print("  Tahap 3 : ML skenario 11 fitur")
        print("  Split    : 60/40, 70/30, 80/20, 90/10")
        print("=" * 64)

        # Selalu ekstrak CSV lengkap 14 fitur.
        df_extracted = run_feature_extraction(
            data_dir=args.data_dir,
            output_csv=args.output_csv,
            station_coords=station_coords,
            valid_labels=args.labels,
            use_fk=True,
        )

        if df_extracted is None or df_extracted.empty:
            print(
                "\n[ERROR] Tidak ada event yang berhasil diekstrak. "
                "Training dibatalkan."
            )
            sys.exit(1)

        if not os.path.exists(args.output_csv):
            print(
                f"[ERROR] Ekstraksi selesai tetapi CSV tidak ditemukan: "
                f"{args.output_csv}"
            )
            sys.exit(1)

        # Satu CSV 14 fitur menjadi sumber kedua skenario.
        run_all_splits(
            output_csv=args.output_csv,
            base_model_output=args.model_output,
            base_output_plots=args.output_plots,
            random_state=args.random_state,
            blind_size=args.blind_size,
            blind_seed=args.blind_seed,
            data_dir=args.data_dir,
            use_fk=True,
        )

    if args.mode in ("predict", "predict_all"):
        if args.pred_input is None:
            print("[ERROR] --pred_input wajib untuk mode predict / predict_all.")
            sys.exit(1)

    if args.mode == "predict":
        # Gunakan --model_output langsung sebagai folder model
        # (tidak perlu ditambah suffix apapun)
        run_prediction(
            pred_input    = args.pred_input,
            model_output  = args.model_output,   # ← langsung, tanpa tambahan suffix
            output_plots  = args.output_plots,
            station_coords = station_coords,
            tag           = os.path.basename(args.model_output.rstrip("/\\")),
            random_state  = args.random_state,
        )

    if args.mode == "predict_all":
        run_all_predictions(
            pred_input        = args.pred_input,
            base_model_output = args.model_output,   # ← folder INDUK
            base_output_plots = args.output_plots,
            station_coords    = station_coords,
            random_state      = args.random_state,
        )

    if args.mode == "blind_test":
        run_blind_test(
            output_csv   = args.output_csv,
            model_output = args.model_output,
            output_plots = args.output_plots,
            blind_size   = args.blind_size,
            random_state = args.blind_seed,  # pakai seed berbeda!
        )

    if args.mode == "blind_new":
        if args.pred_input is None:
            print("[ERROR] --pred_input wajib untuk mode blind_new.")
            sys.exit(1)
        run_blind_test_new_data(
            pred_input     = args.pred_input,
            model_output   = args.model_output,
            output_plots   = args.output_plots,
            station_coords = station_coords,
            has_labels     = args.has_labels,
            already_cut    = args.already_cut,
            random_state   = args.random_state,
        )

    if args.mode == "blind_new_all":
        if args.pred_input is None:
            print("[ERROR] --pred_input wajib untuk mode blind_new_all.")
            sys.exit(1)
        run_blind_new_all(
            pred_input        = args.pred_input,
            base_model_output = args.model_output,
            base_output_plots = args.output_plots,
            station_coords    = station_coords,
            has_labels        = args.has_labels,
            already_cut       = args.already_cut,
            random_state      = args.random_state,
        )

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║               ✅ PIPELINE SELESAI                ║")
    print("╚══════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()

# Langkah 1 — ekstraksi fitur (sekali saja)
# python seismic_classification_new.py --mode extract --data_dir ./dataset_event --output_csv ./output/features.csv

# Langkah 2 — loop semua skenario split (otomatis 60/40, 70/30, 80/20, 90/10)
# python seismic_classification_new.py --mode all_splits --output_csv ./output/features.csv --model_output ./models --output_plots ./plots --data_dir ./dataset_event

# Langkah 2b — loop semua skenario split TANPA fitur FK
# python seismic_classification_new.py --mode all_splits --no_fk --output_csv ./output/features.csv --model_output ./models --output_plots ./plots --data_dir ./dataset_event

# Langkah 3 — prediksi dengan salah satu model split
# python seismic_classification_new.py --mode predict --pred_input ./dataset_predict --model_output ./models/split_60_40 --output_plots ./plots

# Langkah 3b — prediksi dengan SEMUA model split sekaligus
# python seismic_classification_new.py --mode predict_all --pred_input ./dataset_predict --model_output ./models --output_plots ./plots

# Langkah 4 — blind test data baru (belum pernah dilihat model sama sekali)
# python seismic_classification_new.py --mode blind_new --pred_input ./dataset_blind --model_output ./models/split_60_40 --output_plots ./plots --has_labels
