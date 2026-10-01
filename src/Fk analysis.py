"""
FK (Frequency–Wavenumber) analysis for the five-station Merapi seismic array.

Purpose
-------
This module performs array-based FK analysis for validated seismic events
recorded by the five Raspberry Shake/Boom stations used in the study:

    RE5DE, R6940, R265F, R7D17, R0279

The implementation follows the MATLAB workflow used in the research:
    waveform windowing
        -> array geometry
        -> FFT
        -> slowness-grid steering
        -> FK power spectrum
        -> peak slowness
        -> backazimuth
        -> normalized FK/beam power

Outputs
-------
For each event:
    - backazimuth (degrees)
    - slowness (s/km)
    - normalized beam power
    - FK spectrum figure
    - CSV summary

Notes
-----
1. This module assumes that the waveforms have already undergone the
   preprocessing/detection workflow in sta_lta_detection.py, unless the
   user supplies waveform files that are already in the appropriate form.
2. No machine-specific absolute paths are used.
3. The default FK parameters follow the supplied MATLAB code:
       slowness range = -1 to +1 s/km
       grid size       = 100 x 100
       target frequency = 5 Hz
4. The coordinate conversion uses a local Cartesian approximation around
   the array center, as in the MATLAB function latlon2xy().
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import obspy


# ============================================================
# ARRAY CONFIGURATION
# ============================================================

STATIONS = ["RE5DE", "R6940", "R265F", "R7D17", "R0279"]

LAT = np.array(
    [-7.69225, -7.69218, -7.69429, -7.69393, -7.69126],
    dtype=float,
)

LON = np.array(
    [110.43853, 110.44111, 110.43898, 110.44132, 110.44003],
    dtype=float,
)

# Approximate regional earthquake epicenter used in the original
# validation example. This is optional and is NOT used to estimate
# the FK result.
REFERENCE_EPICENTER_LAT = -8.60
REFERENCE_EPICENTER_LON = 111.06


# ============================================================
# FK PARAMETERS
# ============================================================

SLMAX = 1.0          # s/km
NS = 100             # number of grid points per slowness dimension
TARGET_FREQUENCY = 5.0  # Hz

# Event window used in the original example
EVENT_DURATION = 60.0  # seconds

# Assumed reference velocity for optional theoretical comparison
REFERENCE_VELOCITY = 6.0  # km/s


# ============================================================
# GEOMETRY
# ============================================================

def latlon_to_xy(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float,
    lon0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert latitude/longitude to local Cartesian coordinates.

    Parameters
    ----------
    lat, lon : array-like
        Station coordinates in degrees.
    lat0, lon0 : float
        Array-center coordinates in degrees.

    Returns
    -------
    dx, dy : ndarray
        Relative east-west and north-south coordinates in metres.
    """
    earth_radius = 6371000.0

    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)

    dx = earth_radius * dlon * np.cos(np.deg2rad(lat0))
    dy = earth_radius * dlat

    return dx, dy


def array_geometry() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return local array coordinates and array center."""
    lat0 = float(np.mean(LAT))
    lon0 = float(np.mean(LON))

    dx, dy = latlon_to_xy(
        LAT,
        LON,
        lat0,
        lon0,
    )

    return dx, dy, lat0, lon0


# ============================================================
# OPTIONAL REFERENCE AZIMUTH
# ============================================================

def calculate_reference_azimuth(
    lat_epi: float,
    lon_epi: float,
    lat0: float,
    lon0: float,
) -> float:
    """
    Calculate the azimuth from the array center to a reference epicenter.

    This is only for comparison/validation and does not determine the
    observed FK backazimuth.
    """
    dxe, dye = latlon_to_xy(
        np.array([lat_epi]),
        np.array([lon_epi]),
        lat0,
        lon0,
    )

    azimuth = np.degrees(np.arctan2(dxe[0], dye[0]))

    if azimuth < 0:
        azimuth += 360.0

    return float(azimuth)


# ============================================================
# WAVEFORM READING
# ============================================================

def read_event_waveforms(
    waveform_dir: Path,
    event_time: obspy.UTCDateTime,
    duration: float = EVENT_DURATION,
) -> tuple[dict[str, np.ndarray], dict[str, obspy.UTCDateTime]]:
    """
    Read and window the five station waveforms.

    The function searches recursively for MiniSEED files under each
    station directory and extracts the requested event window.

    Parameters
    ----------
    waveform_dir : Path
        Root directory containing station folders.
    event_time : UTCDateTime
        Event origin/onset time in UTC.
    duration : float
        Event window duration in seconds.

    Returns
    -------
    data_all : dict
        Station -> waveform samples.
    start_times : dict
        Station -> actual start time of the selected waveform window.
    """
    data_all = {}
    start_times = {}

    for station in STATIONS:
        station_dir = waveform_dir / station

        if not station_dir.exists():
            print(f"WARNING: station directory not found: {station_dir}")
            continue

        files = sorted(
            [
                p for p in station_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in {".mseed", ".msd", ".seed", ""}
            ]
        )

        if not files:
            print(f"WARNING: no waveform file found for {station}")
            continue

        stream = obspy.Stream()

        # Try candidate files until at least one can be read.
        for file_path in files:
            try:
                stream += obspy.read(str(file_path))
            except Exception:
                continue

        if len(stream) == 0:
            print(f"WARNING: waveform could not be read for {station}")
            continue

        stream.merge(method=1, fill_value=0)

        selected = stream.copy().trim(
            starttime=event_time,
            endtime=event_time + duration,
            pad=False,
        )

        if len(selected) == 0:
            print(f"WARNING: no data in event window for {station}")
            continue

        # Use the first available trace after merging.
        trace = selected[0]

        if trace.stats.npts < 2:
            print(f"WARNING: insufficient samples for {station}")
            continue

        data_all[station] = np.asarray(trace.data, dtype=float)
        start_times[station] = trace.stats.starttime

        print(
            f"{station}: {len(data_all[station])} samples | "
            f"sampling rate = {trace.stats.sampling_rate:.3f} Hz"
        )

    return data_all, start_times


# ============================================================
# ARRAY MATRIX
# ============================================================

def build_array_matrix(
    data_all: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """
    Build a common-length station x sample matrix.

    Only stations available in the input dictionary are retained.
    """
    available = [sta for sta in STATIONS if sta in data_all]

    if len(available) < 2:
        raise ValueError(
            "At least two stations are required for FK analysis."
        )

    n_samples = min(len(data_all[sta]) for sta in available)

    matrix = np.zeros(
        (len(available), n_samples),
        dtype=float,
    )

    for i, station in enumerate(available):
        matrix[i, :] = data_all[station][:n_samples]

    return matrix, available


# ============================================================
# FK ANALYSIS
# ============================================================

def calculate_fk(
    data_matrix: np.ndarray,
    station_names: list[str],
    sampling_rate: float,
    target_frequency: float = TARGET_FREQUENCY,
    slmax: float = SLMAX,
    ns: int = NS,
) -> dict:
    """
    Perform frequency-domain FK analysis.

    Parameters
    ----------
    data_matrix : ndarray
        Array of shape (number_of_stations, number_of_samples).
    station_names : list[str]
        Station names corresponding to matrix rows.
    sampling_rate : float
        Sampling rate in Hz.
    target_frequency : float
        Frequency used for FK evaluation.
    slmax : float
        Maximum absolute slowness in s/km.
    ns : int
        Number of grid points along each slowness dimension.

    Returns
    -------
    dict
        FK spectrum and derived spatial parameters.
    """
    if data_matrix.ndim != 2:
        raise ValueError("data_matrix must be a 2-D array.")

    station_indices = [STATIONS.index(sta) for sta in station_names]

    dx_all, dy_all, _, _ = array_geometry()

    dx = dx_all[station_indices]
    dy = dy_all[station_indices]

    n_samples = data_matrix.shape[1]

    if n_samples < 2:
        raise ValueError("At least two samples are required for FFT.")

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------
    fft_data = np.fft.fft(
        data_matrix,
        axis=1,
    )

    frequencies = np.fft.fftfreq(
        n_samples,
        d=1.0 / sampling_rate,
    )

    # Use the positive-frequency component closest to target.
    positive = frequencies >= 0

    positive_frequencies = frequencies[positive]

    if len(positive_frequencies) == 0:
        raise ValueError("No positive FFT frequencies available.")

    idx_positive = np.argmin(
        np.abs(positive_frequencies - target_frequency)
    )

    idx_frequency = np.flatnonzero(positive)[idx_positive]

    actual_frequency = float(frequencies[idx_frequency])
    omega = 2.0 * np.pi * actual_frequency

    xf = fft_data[:, idx_frequency]

    # --------------------------------------------------------
    # Slowness grid
    # --------------------------------------------------------
    ux = np.linspace(
        -slmax,
        slmax,
        ns,
    )

    uy = np.linspace(
        -slmax,
        slmax,
        ns,
    )

    pfk = np.zeros(
        (ns, ns),
        dtype=float,
    )

    # dx/dy are in metres; slowness is s/km.
    # Convert station coordinates from metres to kilometres.
    dx_km = dx / 1000.0
    dy_km = dy / 1000.0

    for ix, slx in enumerate(ux):
        for iy, sly in enumerate(uy):

            delay = (
                slx * dx_km
                + sly * dy_km
            )

            steering = np.exp(
                -1j * omega * delay
            )

            beam = np.sum(
                xf * np.conj(steering)
            )

            pfk[ix, iy] = np.abs(beam) ** 2

    # --------------------------------------------------------
    # Peak FK
    # --------------------------------------------------------
    max_index = np.unravel_index(
        np.argmax(pfk),
        pfk.shape,
    )

    ix_max, iy_max = max_index

    sx_est = float(ux[ix_max])
    sy_est = float(uy[iy_max])

    slowness_est = float(
        np.sqrt(
            sx_est ** 2
            + sy_est ** 2
        )
    )

    backazimuth_est = float(
        np.degrees(
            np.arctan2(
                sx_est,
                sy_est,
            )
        )
    )

    if backazimuth_est < 0:
        backazimuth_est += 360.0

    # --------------------------------------------------------
    # Normalized FK / beam power
    # --------------------------------------------------------
    # The peak power is normalized to the total FK power so that
    # the value is dimensionless and comparable between events.
    total_power = float(np.sum(pfk))

    if total_power > 0:
        normalized_beam_power = float(
            np.max(pfk) / total_power
        )
    else:
        normalized_beam_power = 0.0

    return {
        "ux": ux,
        "uy": uy,
        "pfk": pfk,
        "target_frequency": target_frequency,
        "actual_frequency": actual_frequency,
        "sx": sx_est,
        "sy": sy_est,
        "slowness": slowness_est,
        "backazimuth": backazimuth_est,
        "peak_power": float(np.max(pfk)),
        "normalized_beam_power": normalized_beam_power,
        "station_names": station_names,
    }


# ============================================================
# FK PLOT
# ============================================================

def plot_fk_spectrum(
    fk_result: dict,
    output_file: Path,
    reference_azimuth: float | None = None,
    reference_velocity: float = REFERENCE_VELOCITY,
) -> None:
    """Plot the FK spectrum and estimated propagation direction."""
    ux = fk_result["ux"]
    uy = fk_result["uy"]
    pfk = fk_result["pfk"]

    sx_est = fk_result["sx"]
    sy_est = fk_result["sy"]

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    # Use logarithmic power for visual contrast while retaining the
    # original FK maximum for numerical feature extraction.
    display_power = 10.0 * np.log10(
        np.maximum(pfk, np.finfo(float).tiny)
    )

    image = ax.imshow(
        display_power.T,
        extent=[
            ux[0],
            ux[-1],
            uy[0],
            uy[-1],
        ],
        origin="lower",
        aspect="auto",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="FK Power (dB)",
    )

    ax.set_xlabel("Slowness X (s/km)")
    ax.set_ylabel("Slowness Y (s/km)")
    ax.set_title("FK Spectrum")

    # Grid
    grid_step = max(1, len(ux) // 10)

    for x in ux[::grid_step]:
        ax.axvline(
            x,
            linestyle=":",
            linewidth=0.5,
            alpha=0.5,
        )

    for y in uy[::grid_step]:
        ax.axhline(
            y,
            linestyle=":",
            linewidth=0.5,
            alpha=0.5,
        )

    # Array center
    ax.plot(
        0,
        0,
        marker="o",
        markersize=9,
        markerfacecolor="yellow",
        markeredgecolor="black",
        label="Array center",
    )

    # FK peak
    ax.plot(
        sx_est,
        sy_est,
        marker="*",
        markersize=14,
        label="FK peak",
    )

    ax.plot(
        [0, sx_est],
        [0, sy_est],
        linewidth=2,
        label="Estimated direction",
    )

    ax.text(
        sx_est,
        sy_est,
        f"  Peak ({sx_est:.2f}, {sy_est:.2f})",
        fontweight="bold",
    )

    # Optional theoretical/reference direction
    if reference_azimuth is not None:
        s_true = 1.0 / reference_velocity

        sx_true = (
            s_true
            * np.sin(np.deg2rad(reference_azimuth))
        )

        sy_true = (
            s_true
            * np.cos(np.deg2rad(reference_azimuth))
        )

        ax.plot(
            [0, sx_true],
            [0, sy_true],
            linestyle="--",
            linewidth=2,
            label="Reference direction",
        )

        ax.plot(
            sx_true,
            sy_true,
            marker="o",
            markersize=8,
        )

    ax.legend(
        loc="best"
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

    print(f"Saved FK figure: {output_file}")


# ============================================================
# SAVE FEATURES
# ============================================================

def save_fk_features(
    fk_result: dict,
    event_time: obspy.UTCDateTime,
    output_file: Path,
) -> None:
    """Save spatial FK features as one CSV row."""
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "event_time_utc",
        "frequency_hz",
        "backazimuth_deg",
        "slowness_s_per_km",
        "normalized_beam_power",
        "peak_power",
        "sx_s_per_km",
        "sy_s_per_km",
        "stations",
    ]

    row = {
        "event_time_utc": event_time.isoformat(),
        "frequency_hz": fk_result["actual_frequency"],
        "backazimuth_deg": fk_result["backazimuth"],
        "slowness_s_per_km": fk_result["slowness"],
        "normalized_beam_power": fk_result["normalized_beam_power"],
        "peak_power": fk_result["peak_power"],
        "sx_s_per_km": fk_result["sx"],
        "sy_s_per_km": fk_result["sy"],
        "stations": ";".join(fk_result["station_names"]),
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

    print(f"Saved FK features: {output_file}")


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def parse_event_time(value: str) -> obspy.UTCDateTime:
    """Convert an ISO-like UTC event time string to UTCDateTime."""
    try:
        return obspy.UTCDateTime(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid UTC event time: {value}"
        ) from exc


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Five-station FK analysis for volcanic seismic events."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing station waveform folders.",
    )

    parser.add_argument(
        "--event-time",
        type=parse_event_time,
        required=True,
        help=(
            "Event time in UTC, e.g. "
            "2023-09-05T18:33:29"
        ),
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=EVENT_DURATION,
        help="Event waveform duration in seconds.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fk"),
        help="Directory for FK figures and feature CSV.",
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
    print("FREQUENCY–WAVENUMBER (FK) ARRAY ANALYSIS")
    print("=" * 70)

    print(f"Event time : {args.event_time}")
    print(f"Duration   : {args.duration:.1f} s")
    print(f"Stations   : {', '.join(STATIONS)}")
    print(f"Target freq: {TARGET_FREQUENCY:.2f} Hz")
    print(
        f"Slowness  : "
        f"{-SLMAX:.2f} to +{SLMAX:.2f} s/km"
    )

    # --------------------------------------------------------
    # Array geometry
    # --------------------------------------------------------
    dx, dy, lat0, lon0 = array_geometry()

    print("\nArray center:")
    print(f"  Latitude  : {lat0:.6f}")
    print(f"  Longitude : {lon0:.6f}")

    for station, x, y in zip(STATIONS, dx, dy):
        print(
            f"  {station}: "
            f"dx={x:.2f} m, dy={y:.2f} m"
        )

    # --------------------------------------------------------
    # Optional reference azimuth
    # --------------------------------------------------------
    reference_azimuth = calculate_reference_azimuth(
        REFERENCE_EPICENTER_LAT,
        REFERENCE_EPICENTER_LON,
        lat0,
        lon0,
    )

    print(
        f"\nReference azimuth from array center: "
        f"{reference_azimuth:.1f} deg"
    )

    # --------------------------------------------------------
    # Read event waveform
    # --------------------------------------------------------
    data_all, start_times = read_event_waveforms(
        waveform_dir=args.data_dir,
        event_time=args.event_time,
        duration=args.duration,
    )

    if len(data_all) < 2:
        raise RuntimeError(
            "Insufficient stations for FK analysis."
        )

    # --------------------------------------------------------
    # Build array matrix
    # --------------------------------------------------------
    data_matrix, station_names = build_array_matrix(
        data_all
    )

    # Sampling rate from the first available trace.
    first_station = station_names[0]
    first_start = start_times[first_station]

    # Re-read first trace metadata to obtain sampling rate.
    first_station_dir = args.data_dir / first_station

    first_files = sorted(
        [
            p for p in first_station_dir.rglob("*")
            if p.is_file()
        ]
    )

    if not first_files:
        raise RuntimeError(
            f"Could not determine sampling rate for {first_station}."
        )

    stream = obspy.read(str(first_files[0]))
    sampling_rate = float(
        stream[0].stats.sampling_rate
    )

    print(
        f"\nArray matrix: "
        f"{data_matrix.shape[0]} stations x "
        f"{data_matrix.shape[1]} samples"
    )

    print(
        f"Sampling rate: "
        f"{sampling_rate:.3f} Hz"
    )

    # --------------------------------------------------------
    # FK
    # --------------------------------------------------------
    fk_result = calculate_fk(
        data_matrix=data_matrix,
        station_names=station_names,
        sampling_rate=sampling_rate,
    )

    print("\nFK result:")
    print(
        f"  Frequency           : "
        f"{fk_result['actual_frequency']:.3f} Hz"
    )
    print(
        f"  Backazimuth         : "
        f"{fk_result['backazimuth']:.3f} deg"
    )
    print(
        f"  Slowness            : "
        f"{fk_result['slowness']:.6f} s/km"
    )
    print(
        f"  Peak FK power       : "
        f"{fk_result['peak_power']:.6e}"
    )
    print(
        f"  Normalized beam power: "
        f"{fk_result['normalized_beam_power']:.6f}"
    )

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------
    figure_file = (
        output_dir
        / f"FK_{args.event_time.strftime('%Y%m%d_%H%M%S')}.png"
    )

    plot_fk_spectrum(
        fk_result=fk_result,
        output_file=figure_file,
        reference_azimuth=reference_azimuth,
    )

    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------
    feature_file = (
        output_dir
        / "fk_spatial_features.csv"
    )

    save_fk_features(
        fk_result=fk_result,
        event_time=args.event_time,
        output_file=feature_file,
    )

    print("\nDONE")


if __name__ == "__main__":
    main()
