"""
STA/LTA-based multi-station seismic event detection.

This module performs:
1. Waveform reading for five seismic stations.
2. Signal preprocessing and instrument-response removal.
3. STA/LTA trigger detection at each station.
4. Multi-station coincidence checking.
5. Event waveform visualization.
6. Saving the detected-event catalogue as CSV.

The detection parameters are kept consistent with the study workflow.
Input waveform paths are supplied through the command line so that the
code does not depend on machine-specific absolute paths.

Example
-------
python src/sta_lta_detection.py \
    --data-dir data/2023 \
    --output-dir results/sta_lta

Expected data structure
-----------------------
data/2023/
├── RE5DE/EHZ.D/AM.RE5DE.00.EHZ.D.2023.238
├── R6940/EHZ.D/AM.R6940.00.EHZ.D.2023.238
├── R265F/EHZ.D/AM.R265F.00.EHZ.D.2023.238
├── R7D17/EHZ.D/AM.R7D17.00.EHZ.D.2023.238
└── R0279/EHZ.D/AM.R0279.00.EHZ.D.2023.238
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import obspy
from matplotlib.lines import Line2D
from obspy.core.util.attribdict import AttribDict
from obspy.signal.trigger import classic_sta_lta, trigger_onset


# ============================================================
# STUDY CONFIGURATION
# ============================================================

STATIONS = {
    "RE5DE": Path("RE5DE/EHZ.D/AM.RE5DE.00.EHZ.D.2023.238"),
    "R6940": Path("R6940/EHZ.D/AM.R6940.00.EHZ.D.2023.238"),
    "R265F": Path("R265F/EHZ.D/AM.R265F.00.EHZ.D.2023.238"),
    "R7D17": Path("R7D17/EHZ.D/AM.R7D17.00.EHZ.D.2023.238"),
    "R0279": Path("R0279/EHZ.D/AM.R0279.00.EHZ.D.2023.238"),
}

PAZ = AttribDict(
    {
        "poles": [
            -1 + 3.03j,
            -1 - 3.03j,
            -3.03 + 666.67j,
            -3.03 - 666.67j,
        ],
        "zeros": [0j, 0j, 0j],
        "sensitivity": 360000000,
        "gain": 693.0,
    }
)

# STA/LTA parameters
STA_SEC = 10
LTA_SEC = 120
ON_TRIG = 3.0
OFF_TRIG = 0.5

# Multi-station coincidence parameters
COINCIDENCE_WINDOW = 20
MIN_STATION = 5

# Visualization window around detected event
EVENT_WINDOW = 120

# Instrument-response removal and filtering
PRE_FILTER = [0.5, 0.7, 10.0, 12.0]
WATER_LEVEL = 60
BANDPASS_FREQMIN = 0.8
BANDPASS_FREQMAX = 1.8
BANDPASS_CORNERS = 4


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_trace(trace: obspy.Trace) -> obspy.Trace:
    """
    Apply the preprocessing used before STA/LTA detection.

    Processing:
    - demean
    - linear detrend
    - 5% taper
    - instrument-response removal
    - bandpass filtering (0.8-1.8 Hz)
    - conversion to float64

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform trace.

    Returns
    -------
    obspy.Trace
        Preprocessed waveform trace.
    """
    tr = trace.copy()

    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper(max_percentage=0.05)

    tr.simulate(
        paz_remove=PAZ,
        pre_filt=PRE_FILTER,
        water_level=WATER_LEVEL,
    )

    tr.filter(
        "bandpass",
        freqmin=BANDPASS_FREQMIN,
        freqmax=BANDPASS_FREQMAX,
        corners=BANDPASS_CORNERS,
        zerophase=True,
    )

    tr.data = np.asarray(tr.data, dtype=np.float64)

    return tr


# ============================================================
# STA/LTA DETECTION
# ============================================================

def detect_station_events(trace: obspy.Trace) -> list[tuple[obspy.UTCDateTime, obspy.UTCDateTime]]:
    """
    Detect seismic triggers for one station using classic STA/LTA.

    Parameters
    ----------
    trace : obspy.Trace
        Preprocessed waveform.

    Returns
    -------
    list of tuple
        Each tuple contains (onset_time, offset_time).
    """
    fs = trace.stats.sampling_rate

    nsta = int(STA_SEC * fs)
    nlta = int(LTA_SEC * fs)

    if nsta <= 0 or nlta <= nsta:
        raise ValueError(
            f"Invalid STA/LTA window: STA={nsta} samples, LTA={nlta} samples."
        )

    cft = classic_sta_lta(
        np.abs(trace.data),
        nsta,
        nlta,
    )

    triggers = trigger_onset(
        cft,
        ON_TRIG,
        OFF_TRIG,
    )

    picks = []

    for on, off in triggers:
        t_on = trace.stats.starttime + on / fs
        t_off = trace.stats.starttime + off / fs
        picks.append((t_on, t_off))

    return picks


# ============================================================
# MULTI-STATION COINCIDENCE
# ============================================================

def find_multi_station_events(
    all_picks: dict[str, list[tuple[obspy.UTCDateTime, obspy.UTCDateTime]]]
) -> list[tuple[obspy.UTCDateTime, list[str]]]:
    """
    Identify events detected coherently across multiple stations.

    An event is retained when at least MIN_STATION stations have an
    STA/LTA onset within COINCIDENCE_WINDOW seconds of the candidate time.
    """
    candidate_times = []

    for picks in all_picks.values():
        for t_on, _ in picks:
            candidate_times.append(t_on)

    candidate_times.sort()

    final_events = []

    for t in candidate_times:
        stations = []

        for sta, picks in all_picks.items():
            for t_on, _ in picks:
                if abs(t - t_on) <= COINCIDENCE_WINDOW:
                    stations.append(sta)
                    break

        if len(stations) >= MIN_STATION:
            if (
                not final_events
                or abs(t - final_events[-1][0]) > COINCIDENCE_WINDOW
            ):
                final_events.append((t, stations))

    return final_events


# ============================================================
# SAVE EVENT CATALOGUE
# ============================================================

def save_event_catalogue(
    events: list[tuple[obspy.UTCDateTime, list[str]]],
    output_file: Path,
) -> None:
    """Save detected multi-station events as a CSV file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "event_id",
                "event_time_utc",
                "n_stations",
                "stations",
            ]
        )

        for i, (event_time, stations) in enumerate(events, start=1):
            writer.writerow(
                [
                    f"Event_{i:03d}",
                    event_time.isoformat(),
                    len(stations),
                    ";".join(stations),
                ]
            )


# ============================================================
# EVENT VISUALIZATION
# ============================================================

def plot_detected_event(
    event_id: int,
    event_time: obspy.UTCDateTime,
    stations: list[str],
    all_traces: dict[str, obspy.Trace],
    all_picks: dict[str, list[tuple[obspy.UTCDateTime, obspy.UTCDateTime]]],
    output_dir: Path,
) -> None:
    """Plot and save the waveform of one detected multi-station event."""
    fig, axes = plt.subplots(
        len(all_traces),
        1,
        figsize=(14, 10),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    fig.suptitle(
        f"Multi-Station Event {event_id:03d} | "
        f"{event_time.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
        f"{len(stations)} Stations",
        fontsize=14,
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="red",
            ls="--",
            lw=1.5,
            label="STA/LTA Onset",
        ),
        Line2D(
            [0],
            [0],
            color="blue",
            ls="--",
            lw=1.5,
            label="STA/LTA Offset",
        ),
    ]

    t0 = event_time - EVENT_WINDOW / 2
    t1 = event_time + EVENT_WINDOW / 2

    for ax, (sta, tr) in zip(axes, all_traces.items()):
        tr_cut = tr.copy().trim(t0, t1)

        if len(tr_cut.data) == 0:
            continue

        data = tr_cut.data
        times = tr_cut.times()

        ax.plot(
            times,
            data,
            color="black",
            lw=0.8,
        )

        ax.set_ylabel(
            f"{sta}\nVelocity\n(m/s)",
            fontsize=9,
        )

        for t_on, t_off in all_picks[sta]:
            if t0 <= t_on <= t1:
                ax.axvline(
                    t_on - t0,
                    color="red",
                    ls="--",
                    lw=1.2,
                )

                ax.axvline(
                    t_off - t0,
                    color="blue",
                    ls="--",
                    lw=1.2,
                )

        ax.grid(alpha=0.3)

    fig.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=10,
        frameon=True,
    )

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        output_dir
        / f"Event_{event_id:03d}_{event_time.strftime('%H%M%S')}.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {filename}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_detection(data_dir: Path, output_dir: Path) -> None:
    """
    Run the complete STA/LTA multi-station detection workflow.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_picks = {}
    all_traces = {}

    print("=" * 65)
    print("STA/LTA MULTI-STATION SEISMIC EVENT DETECTION")
    print("=" * 65)

    for sta, relative_path in STATIONS.items():
        waveform_path = data_dir / relative_path

        print(f"\nProcessing station: {sta}")
        print(f"Input: {waveform_path}")

        if not waveform_path.exists():
            raise FileNotFoundError(
                f"Waveform file not found for {sta}: {waveform_path}"
            )

        stream = obspy.read(str(waveform_path))

        if len(stream) == 0:
            raise ValueError(f"No waveform trace found for station {sta}.")

        tr = stream[0]

        tr = preprocess_trace(tr)

        picks = detect_station_events(tr)

        all_picks[sta] = picks
        all_traces[sta] = tr

        print(f"{sta}: {len(picks)} events detected")

    final_events = find_multi_station_events(all_picks)

    print("\n" + "=" * 65)
    print(f"TOTAL MULTI-STATION EVENTS: {len(final_events)}")
    print("=" * 65)

    catalogue_file = output_dir / "sta_lta_event_catalogue.csv"
    save_event_catalogue(final_events, catalogue_file)

    print(f"Event catalogue saved: {catalogue_file}")

    figure_dir = output_dir / "figures"

    for i, (event_time, stations) in enumerate(final_events, start=1):
        plot_detected_event(
            event_id=i,
            event_time=event_time,
            stations=stations,
            all_traces=all_traces,
            all_picks=all_picks,
            output_dir=figure_dir,
        )

    print("\nDONE")


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="STA/LTA-based multi-station seismic event detection."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing the station waveform folders.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sta_lta"),
        help="Directory for event catalogue and figures.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_detection(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
