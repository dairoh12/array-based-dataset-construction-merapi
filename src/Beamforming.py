"""
Array beamforming / array-processing analysis for the five-station
Merapi seismic array.

This module is based on the ObsPy array_processing workflow used in the
study. It estimates propagation parameters from the five-station array:

    - back azimuth (degrees)
    - slowness (s/km)
    - relative beam power
    - absolute beam power
    - peak beam-power histogram

The analysis uses ObsPy's array_processing() implementation with the
parameter configuration used in the research workflow.

Stations
--------
RE5DE, R6940, R265F, R7D17, R0279

Frequency band
--------------
0.8-1.8 Hz

Slowness search
---------------
-3 to +3 s/km with 0.03 s/km spacing.

This file does not contain machine-specific absolute paths. Input data
are supplied through the command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import obspy

from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential
from obspy.signal.array_analysis import array_processing
from obspy.signal.invsim import corn_freq_2_paz


# ============================================================
# ARRAY CONFIGURATION
# ============================================================

STATIONS = [
    {
        "name": "RE5DE",
        "latitude": -7.69225,
        "longitude": 110.43853,
        "elevation": 0.450,
    },
    {
        "name": "R6940",
        "latitude": -7.69218,
        "longitude": 110.44111,
        "elevation": 0.450,
    },
    {
        "name": "R265F",
        "latitude": -7.69429,
        "longitude": 110.43898,
        "elevation": 0.450,
    },
    {
        "name": "R7D17",
        "latitude": -7.69393,
        "longitude": 110.44132,
        "elevation": 0.450,
    },
    {
        "name": "R0279",
        "latitude": -7.69126,
        "longitude": 110.44003,
        "elevation": 0.450,
    },
]


# Instrument response parameters used in the original workflow.
PAZ = AttribDict(
    {
        "poles": [
            -1 - 3.03j,
            -3.03 - 666.67j,
        ],
        "zeros": [
            0j,
            0j,
            0j,
        ],
        "sensitivity": 360000000,
        "gain": 693.0,
    }
)


# ============================================================
# ARRAY-PROCESSING PARAMETERS
# ============================================================

SLL_X = -3.0
SLM_X = 3.0
SLL_Y = -3.0
SLM_Y = 3.0
SL_S = 0.03

WIN_LEN = 1.0
WIN_FRAC = 0.5

FRQ_LOW = 0.8
FRQ_HIGH = 1.8

PREWHITEN = 0

SEMB_THRES = -1e9
VEL_THRES = -1e9

# Histogram configuration used in the original analysis.
N_BAZ = 30
N_SLOW = 45
SLOW_MAX = 3.0


# ============================================================
# PAZ / TRACE PREPARATION
# ============================================================

def attach_station_metadata(stream: obspy.Stream) -> obspy.Stream:
    """
    Attach station coordinates and PAZ information to the traces.

    Traces are assigned to the five stations according to the order
    in the input stream. The function first attempts to identify a
    station from trace.stats.station; if unavailable, the original
    trace order is used.
    """
    result = stream.copy()

    # First try station-code matching.
    assigned = set()

    for trace in result:
        code = str(
            getattr(trace.stats, "station", "")
        ).strip()

        match = next(
            (
                station
                for station in STATIONS
                if station["name"] == code
                and station["name"] not in assigned
            ),
            None,
        )

        if match is not None:
            trace.stats.paz = PAZ.copy()
            trace.stats.coordinates = AttribDict(
                {
                    "latitude": match["latitude"],
                    "longitude": match["longitude"],
                    "elevation": match["elevation"],
                }
            )
            assigned.add(match["name"])

    # For streams without recognizable station metadata, use order.
    unmatched = [
        station
        for station in STATIONS
        if station["name"] not in assigned
    ]

    for trace in result:
        if hasattr(trace.stats, "coordinates"):
            continue

        if not unmatched:
            break

        station = unmatched.pop(0)

        trace.stats.paz = PAZ.copy()
        trace.stats.coordinates = AttribDict(
            {
                "latitude": station["latitude"],
                "longitude": station["longitude"],
                "elevation": station["elevation"],
            }
        )

    return result


def correct_instrument_response(
    stream: obspy.Stream,
) -> obspy.Stream:
    """
    Simulate a 1-Hz corner-frequency response.

    This follows the original workflow:
        paz1hz = corn_freq_2_paz(1.0, damp=0.707)
        st.simulate(paz_remove='self', paz_simulate=paz1hz)
    """
    st = stream.copy()

    paz1hz = corn_freq_2_paz(
        1.0,
        damp=0.707,
    )

    st.simulate(
        paz_remove="self",
        paz_simulate=paz1hz,
    )

    return st


# ============================================================
# TIME WINDOW
# ============================================================

def common_time_window(
    stream: obspy.Stream,
) -> tuple[obspy.UTCDateTime, obspy.UTCDateTime]:
    """Return the latest start and earliest end among all traces."""
    if len(stream) == 0:
        raise ValueError("The input stream is empty.")

    start_time = max(
        trace.stats.starttime
        for trace in stream
    )

    end_time = min(
        trace.stats.endtime
        for trace in stream
    )

    if start_time >= end_time:
        raise ValueError(
            "No common time interval exists among the traces."
        )

    return start_time, end_time


# ============================================================
# ARRAY PROCESSING
# ============================================================

def run_array_processing(
    stream: obspy.Stream,
) -> np.ndarray:
    """
    Run ObsPy array_processing using the study parameters.

    Returns
    -------
    ndarray
        Columns:
        time, relative power, absolute power,
        back azimuth, slowness.
    """
    start_time, end_time = common_time_window(stream)

    print(f"Adjusted start time: {start_time}")
    print(f"Adjusted end time  : {end_time}")

    kwargs = dict(
        sll_x=SLL_X,
        slm_x=SLM_X,
        sll_y=SLL_Y,
        slm_y=SLM_Y,
        sl_s=SL_S,
        win_len=WIN_LEN,
        win_frac=WIN_FRAC,
        frqlow=FRQ_LOW,
        frqhigh=FRQ_HIGH,
        prewhiten=PREWHITEN,
        semb_thres=SEMB_THRES,
        vel_thres=VEL_THRES,
        stime=start_time,
        etime=end_time,
    )

    out = array_processing(
        stream,
        **kwargs,
    )

    if out.size == 0:
        raise RuntimeError(
            "ObsPy array_processing returned no results."
        )

    return out


# ============================================================
# RESULT EXTRACTION
# ============================================================

def normalize_backazimuth(
    backazimuth: np.ndarray,
) -> np.ndarray:
    """Convert backazimuth values to the 0-360 degree range."""
    baz = np.asarray(
        backazimuth,
        dtype=float,
    ).copy()

    baz[baz < 0.0] += 360.0

    return baz


def find_best_window(
    output: np.ndarray,
) -> dict:
    """
    Find the array-processing window with maximum relative power.
    """
    if output.ndim != 2 or output.shape[1] < 5:
        raise ValueError(
            "Unexpected array_processing output shape."
        )

    idx = int(
        np.argmax(output[:, 1])
    )

    baz = float(
        output[idx, 3] % 360.0
    )

    slow = float(
        output[idx, 4]
    )

    return {
        "index": idx,
        "window": idx + 1,
        "backazimuth_deg": baz,
        "slowness_s_per_km": slow,
        "relative_power": float(output[idx, 1]),
        "absolute_power": float(output[idx, 2]),
        "time": float(output[idx, 0]),
    }


# ============================================================
# BEAM-POWER HISTOGRAM
# ============================================================

def calculate_beam_power_histogram(
    output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate relative beam power into backazimuth/slowness bins.

    Returns
    -------
    hist, baz_edges, slow_edges
    """
    baz = normalize_backazimuth(
        output[:, 3]
    )

    slow = output[:, 4]
    rel_power = output[:, 1]

    abins = (
        np.arange(N_BAZ + 1)
        * 360.0
        / N_BAZ
    )

    sbins = np.linspace(
        0,
        SLOW_MAX,
        N_SLOW + 1,
    )

    hist, baz_edges, slow_edges = np.histogram2d(
        baz,
        slow,
        bins=[
            abins,
            sbins,
        ],
        weights=rel_power,
    )

    return hist, baz_edges, slow_edges


def find_histogram_peak(
    hist: np.ndarray,
    baz_edges: np.ndarray,
    slow_edges: np.ndarray,
) -> dict:
    """Return the maximum beam-power histogram bin."""
    if not np.any(hist):
        return {
            "beam_power": 0.0,
            "backazimuth_min": np.nan,
            "backazimuth_max": np.nan,
            "slowness_min": np.nan,
            "slowness_max": np.nan,
        }

    imax, jmax = np.unravel_index(
        np.argmax(hist),
        hist.shape,
    )

    return {
        "beam_power": float(
            hist[imax, jmax]
        ),
        "backazimuth_min": float(
            baz_edges[imax]
        ),
        "backazimuth_max": float(
            baz_edges[imax + 1]
        ),
        "slowness_min": float(
            slow_edges[jmax]
        ),
        "slowness_max": float(
            slow_edges[jmax + 1]
        ),
    }


# ============================================================
# POLAR BEAMFORMING PLOT
# ============================================================

def plot_beamforming_result(
    output: np.ndarray,
    hist: np.ndarray,
    baz_edges: np.ndarray,
    slow_edges: np.ndarray,
    output_file: Path,
    title: str = "Beamforming Result",
) -> None:
    """Create and save the polar backazimuth-slowness beam-power plot."""
    cmap = obspy_sequential

    baz_edges_rad = np.radians(
        baz_edges
    )

    dh = abs(
        slow_edges[1]
        - slow_edges[0]
    )

    dw = abs(
        baz_edges_rad[1]
        - baz_edges_rad[0]
    )

    fig = plt.figure(
        figsize=(8, 8)
    )

    cax = fig.add_axes(
        [0.85, 0.20, 0.05, 0.50]
    )

    ax = fig.add_axes(
        [0.10, 0.10, 0.70, 0.70],
        polar=True,
    )

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")

    hist_max = (
        float(np.max(hist))
        if hist.size and np.max(hist) > 0
        else 1.0
    )

    for i, row in enumerate(hist):
        normalized_row = row / hist_max

        ax.bar(
            (i * dw) * np.ones(N_SLOW),
            height=dh * np.ones(N_SLOW),
            width=dw,
            bottom=dh * np.arange(N_SLOW),
            color=cmap(normalized_row),
            align="edge",
        )

    ax.set_xticks(
        np.linspace(
            0,
            2 * np.pi,
            4,
            endpoint=False,
        )
    )

    ax.set_xticklabels(
        ["N", "E", "S", "W"]
    )

    ax.set_ylim(
        0,
        SLOW_MAX,
    )

    for label in ax.get_yticklabels():
        label.set_color("grey")

    ColorbarBase(
        cax,
        cmap=cmap,
        norm=Normalize(
            vmin=float(hist.min())
            if hist.size
            else 0.0,
            vmax=float(hist.max())
            if hist.size and hist.max() > 0
            else 1.0,
        ),
    )

    cax.set_ylabel(
        "Relative beam power",
        rotation=90,
    )

    ax.set_title(
        title,
        pad=20,
        fontweight="bold",
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved beamforming figure: {output_file}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    best: dict,
    histogram_peak: dict,
    output_file: Path,
) -> None:
    """Save beamforming features to a CSV file."""
    import csv

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "window",
        "time",
        "backazimuth_deg",
        "slowness_s_per_km",
        "relative_power",
        "absolute_power",
        "histogram_beam_power",
        "histogram_backazimuth_min_deg",
        "histogram_backazimuth_max_deg",
        "histogram_slowness_min_s_per_km",
        "histogram_slowness_max_s_per_km",
    ]

    row = {
        "window": best["window"],
        "time": best["time"],
        "backazimuth_deg": best["backazimuth_deg"],
        "slowness_s_per_km": best["slowness_s_per_km"],
        "relative_power": best["relative_power"],
        "absolute_power": best["absolute_power"],
        "histogram_beam_power": histogram_peak[
            "beam_power"
        ],
        "histogram_backazimuth_min_deg": histogram_peak[
            "backazimuth_min"
        ],
        "histogram_backazimuth_max_deg": histogram_peak[
            "backazimuth_max"
        ],
        "histogram_slowness_min_s_per_km": histogram_peak[
            "slowness_min"
        ],
        "histogram_slowness_max_s_per_km": histogram_peak[
            "slowness_max"
        ],
    }

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerow(row)

    print(
        f"Saved beamforming results: {output_file}"
    )


# ============================================================
# INPUT STREAM
# ============================================================

def read_input_stream(
    input_file: Path,
) -> obspy.Stream:
    """Read a MiniSEED waveform file."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input MiniSEED file not found: {input_file}"
        )

    stream = obspy.read(
        str(input_file)
    )

    if len(stream) < 2:
        raise ValueError(
            "At least two station traces are required."
        )

    return stream


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Five-station ObsPy array beamforming / "
            "array-processing analysis."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Input MiniSEED file containing the "
            "multi-station event traces."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/beamforming"
        ),
        help=(
            "Directory for beamforming figure "
            "and CSV results."
        ),
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Beamforming Result",
        help="Figure title.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_arguments()

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "FIVE-STATION ARRAY BEAMFORMING / ARRAY PROCESSING"
    )
    print("=" * 70)

    print(
        f"Input file : {args.input}"
    )
    print(
        f"Frequency  : {FRQ_LOW:.1f}-{FRQ_HIGH:.1f} Hz"
    )
    print(
        f"Slowness   : "
        f"{SLL_X:.2f} to {SLM_X:.2f} s/km"
    )
    print(
        f"Slowness step: {SL_S:.2f} s/km"
    )
    print(
        f"Window     : {WIN_LEN:.1f} s"
    )
    print(
        f"Window step fraction: {WIN_FRAC:.2f}"
    )

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------
    stream = read_input_stream(
        args.input
    )

    print(
        f"\nNumber of input traces: {len(stream)}"
    )

    # --------------------------------------------------------
    # Attach station metadata
    # --------------------------------------------------------
    stream = attach_station_metadata(
        stream
    )

    # --------------------------------------------------------
    # Instrument correction
    # --------------------------------------------------------
    stream = correct_instrument_response(
        stream
    )

    # --------------------------------------------------------
    # Array processing
    # --------------------------------------------------------
    output = run_array_processing(
        stream
    )

    # --------------------------------------------------------
    # Normalize backazimuth
    # --------------------------------------------------------
    output = output.copy()
    output[:, 3] = normalize_backazimuth(
        output[:, 3]
    )

    # --------------------------------------------------------
    # Best window
    # --------------------------------------------------------
    best = find_best_window(
        output
    )

    print("\n" + "=" * 60)
    print("BEST ARRAY-PROCESSING WINDOW")
    print("=" * 60)
    print(
        f"Window         : {best['window']}"
    )
    print(
        f"Back azimuth   : "
        f"{best['backazimuth_deg']:.2f}°"
    )
    print(
        f"Slowness       : "
        f"{best['slowness_s_per_km']:.3f} s/km"
    )
    print(
        f"Relative power : "
        f"{best['relative_power']:.6f}"
    )
    print(
        f"Absolute power : "
        f"{best['absolute_power']:.6f}"
    )

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------
    hist, baz_edges, slow_edges = (
        calculate_beam_power_histogram(
            output
        )
    )

    histogram_peak = find_histogram_peak(
        hist,
        baz_edges,
        slow_edges,
    )

    print("\n" + "=" * 60)
    print("BEAM POWER HISTOGRAM PEAK")
    print("=" * 60)
    print(
        f"Beam power      : "
        f"{histogram_peak['beam_power']:.6f}"
    )
    print(
        f"Back azimuth bin: "
        f"{histogram_peak['backazimuth_min']:.1f} - "
        f"{histogram_peak['backazimuth_max']:.1f}°"
    )
    print(
        f"Slowness bin    : "
        f"{histogram_peak['slowness_min']:.3f} - "
        f"{histogram_peak['slowness_max']:.3f} s/km"
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    figure_file = (
        output_dir
        / "beamforming_result.png"
    )

    plot_beamforming_result(
        output=output,
        hist=hist,
        baz_edges=baz_edges,
        slow_edges=slow_edges,
        output_file=figure_file,
        title=args.title,
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------
    csv_file = (
        output_dir
        / "beamforming_features.csv"
    )

    save_results(
        best=best,
        histogram_peak=histogram_peak,
        output_file=csv_file,
    )

    print("\nDONE")


if __name__ == "__main__":
    main()
