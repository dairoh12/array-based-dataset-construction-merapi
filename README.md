# Array-Based Dataset Construction for Machine Learning Classification of Long-Distance Volcanic Seismic Signals

This repository contains the source code and documentation associated with the study:

**Array-Based Dataset Construction for Machine Learning Classification of Long-Distance Volcanic Seismic Signals**

The study presents an array-based computational framework for constructing a labelled seismic-event dataset from continuous recordings acquired by a remote seismic array at Mount Merapi, Indonesia. The framework integrates automatic candidate-event detection, array processing, catalogue-based validation, feature extraction, and machine-learning classification.

---

## 1. Study Overview

Seismic-event classification using recordings acquired at stations located far from a volcanic source is challenging because seismic waves undergo attenuation and are increasingly affected by environmental and instrumental noise.

This study develops a computational workflow for transforming continuous seismic recordings from a remote five-station seismic array into a labelled dataset suitable for supervised machine-learning classification.

The workflow combines:

1. seismic waveform preprocessing;
2. Short-Term Average/Long-Term Average (STA/LTA) candidate-event detection;
3. array-based beamforming;
4. Frequency–Wavenumber (FK) analysis;
5. catalogue matching and event validation;
6. manual waveform quality control;
7. temporal and spectral feature extraction; and
8. machine-learning classification using SVM-RBF and XGBoost.

The resulting dataset contains **275 validated seismic events** belonging to four classes:

- Multiphase (MP)
- Rockfall (RF)
- Volcano-Tectonic B (VTB)
- Regional Earthquake (RE)

The final class distribution is:

| Class | Number of events |
|---|---:|
| Multiphase | 71 |
| Regional Earthquake | 49 |
| Rockfall | 84 |
| Volcano-Tectonic B (VTB) | 71 |
| **Total** | **275** |

---

## 2. Study Site and Data

The study uses continuous seismic recordings from the Universitas Gadjah Mada (UGM) remote seismic array at Mount Merapi, Indonesia.

The recordings used in the study cover:

**16 August – 12 September 2023**

The observation period corresponds to approximately 28 days of continuous recording.

The array consists of five seismic stations:

| Station | Instrument code |
|---|---|
| UGM1 | RE5DE |
| UGM2 | R6940 |
| UGM3 | R265F |
| UGM4 | R7D17 |
| UGM5 | R0279 |

The array is located approximately 16 km south of the Mount Merapi summit. The five stations are arranged in a pentagonal configuration with an average inter-station distance of approximately 250 m.

All stations record the vertical component at a sampling frequency of 100 Hz.

The original waveform data are **not included in this repository** because access to the seismic recordings is subject to data-access restrictions.

---

## 3. Computational Workflow

The complete computational workflow is organized into the following stages:

```text
Continuous seismic recordings
            |
            v
    Waveform preprocessing
            |
            v
       STA/LTA detection
            |
            v
     Candidate events
            |
            v
       Array processing
       /              \
      v                v
 Beamforming           FK analysis
                       |
                Back-azimuth
                  Slowness
            |
            v
   Catalogue matching
   /                \
  v                  v
BMKG catalogue    BPPTKG catalogue
   \                /
    \              /
     v            v
     Event validation
            |
            v
   Manual quality control
            |
            v
     Validated events
            |
            v
    Feature extraction
            |
            v
  Temporal + spectral features
            |
            v
 Machine-learning classification
       /              \
      v                v
   SVM-RBF          XGBoost
