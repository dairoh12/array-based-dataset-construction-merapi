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

| Station |
|---|
| RE5DE |
| R6940 |
| R265F |
| R7D17 |
| R0279 |

The array is located approximately 16 km south of the Mount Merapi summit. The five stations are arranged in a pentagonal configuration with an average inter-station distance of approximately 250 m.

All stations record the vertical component at a sampling frequency of 100 Hz.

The original waveform data are **not included in this repository** because access to the seismic recordings is subject to data-access restrictions.

---

## 3. Computational Workflow

The complete computational workflow is organized into the following stages:

1. Continuous seismic recordings
2. Waveform preprocessing
3. STA/LTA candidate-event detection
4. Candidate-event screening
5. Beamforming
6. FK analysis
7. Catalogue matching
8. Event validation
9. Manual quality control
10. Validated seismic events
11. Temporal and spectral feature extraction
12. Machine-learning classification using SVM-RBF and XGBoost

Beamforming and FK analysis are used during the **dataset-construction and validation stages**. They are not used as machine-learning input features in the classification stage of this study.

---

## 4. Waveform Preprocessing

The preprocessing workflow includes:

1. instrument-response correction using a Pole–Zero (PAZ) model;
2. cosine pre-filtering during response correction;
3. removal of the mean;
4. linear detrending;
5. merging of waveform segments;
6. tapering using a 5% Hann window; and
7. fourth-order zero-phase Butterworth band-pass filtering.

The main frequency band used in the dataset-construction workflow is:

**0.8–1.8 Hz**

The preprocessing parameters implemented in the source code should be considered together with the configuration files and scripts included in this repository.

---

## 5. Candidate Event Detection

Candidate seismic events are initially detected from the continuous recordings using the Short-Term Average/Long-Term Average (STA/LTA) method.

STA/LTA detection is used as an initial screening procedure to identify time intervals showing significant changes in seismic energy.

Importantly, STA/LTA detections are treated as **candidate events** rather than automatically identified seismic events.

Each candidate is subsequently evaluated using array processing and catalogue validation.

---

## 6. Array-Based Evaluation

Each candidate event is evaluated using the spatial coherence and propagation characteristics observed across the five stations.

### 6.1 Beamforming

Beamforming is used to combine signals recorded at multiple stations while accounting for differences in arrival time.

The purpose of beamforming in this workflow is to:

- enhance coherent seismic signals;
- suppress relatively incoherent components;
- evaluate signal coherence across the array; and
- determine the maximum beam power associated with the dominant arrival.

### 6.2 Frequency–Wavenumber Analysis

Frequency–Wavenumber (FK) analysis is used to estimate propagation characteristics, particularly:

- back azimuth; and
- slowness.

The array-derived propagation parameters are subsequently used as supporting information during event validation.

---

## 7. Catalogue Validation and Event Labelling

Candidate events are compared with two independent reference catalogues.

### 7.1 BMKG

The Meteorology, Climatology, and Geophysics Agency (BMKG) earthquake catalogue is used to identify and validate regional earthquake events.

### 7.2 BPPTKG

The Center for Research and Development of Geological Disaster Technology (BPPTKG) volcanic event catalogue is used as a reference for events associated with Mount Merapi activity.

Catalogue matching is performed using event timing and the corresponding matching criteria implemented in the workflow.

Candidates that satisfy the array-processing criteria and catalogue-validation requirements are retained for the final labelled dataset.

A final manual waveform-quality-control step is also performed to remove events that cannot be adequately verified or show ambiguous characteristics.

---

## 8. Dataset Construction

The initial STA/LTA detection produced:

**446 candidate events**

After array-based evaluation, catalogue matching, and manual quality control:

- **275 validated events**
- **171 rejected or ambiguous candidates**

Therefore, the dataset construction can be summarized as:

**446 candidates → 275 validated events + 171 rejected/ambiguous candidates**

The final dataset contains four seismic-event classes:

- Multiphase
- Regional Earthquake
- Rockfall
- Volcano-Tectonic B

The dataset is subsequently divided using stratified train–test splitting.

The evaluated train–test scenarios are:

- 60:40
- 70:30
- 80:20
- 90:10

The corresponding total training and testing sample sizes are:

| Split | Training events | Testing events |
|---|---:|---:|
| 60:40 | 165 | 110 |
| 70:30 | 192 | 83 |
| 80:20 | 220 | 55 |
| 90:10 | 247 | 28 |

---

## 9. Feature Extraction

A total of **11 temporal and spectral features** are extracted from each validated event.

### 9.1 Temporal Features

1. Mean
2. Standard deviation
3. Skewness
4. Kurtosis
5. Maximum amplitude
6. Zero crossing

### 9.2 Spectral Features

7. Dominant frequency
8. Energy centre
9. RMS bandwidth
10. Spectral centroid
11. Spectral entropy

These features represent complementary characteristics of waveform amplitude, temporal shape, and frequency-domain energy distribution.

---

## 10. Machine-Learning Models

Two supervised machine-learning algorithms are evaluated.

### 10.1 SVM-RBF

Support Vector Machine with a Radial Basis Function kernel is used as a kernel-based classification model.

### 10.2 XGBoost

Extreme Gradient Boosting is used as a tree-based ensemble classification model.

The two approaches provide complementary modelling strategies for evaluating the classification of the extracted temporal and spectral features.

---

## 11. Model Training and Hyperparameter Optimisation

Hyperparameter optimisation is performed using:

`GridSearchCV`

with:

`Stratified 5-fold cross-validation`

and:

`Macro-F1`

as the optimisation criterion.

The machine-learning pipeline incorporates preprocessing and class-balancing operations within the training procedure.

Standardisation and SMOTE are fitted only on the training data within the cross-validation process to prevent information from the test data from being used during model training.

The detailed parameter search spaces used in the study are provided in the corresponding source code and configuration files.

---

## 12. Evaluation Metrics

The classification models are evaluated using:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- Macro-F1
- Weighted-F1
- Confusion matrix
- Receiver Operating Characteristic (ROC) curves
- Area Under the Curve (AUC)
- Bootstrap-based 95% confidence intervals
- Learning curves

Macro-averaged metrics are used to evaluate performance across all classes without weighting classes according to their sample size.

Weighted-F1 accounts for the relative number of samples in each class.

---

## 13. Principal Component Analysis

Principal Component Analysis (PCA) is used as an exploratory analysis of the feature space.

PCA is used to investigate:

- feature redundancy;
- feature structure;
- variance distribution; and
- relationships among the extracted features.

PCA is **not used as an input transformation within the machine-learning classification pipeline**.

---

## 14. Repository Structure

The repository is organized as follows:

`array-based-dataset-construction-merapi/`

- `README.md`
- `LICENSE`
- `CITATION.cff`
- `requirements.txt`
- `src/`
  - `preprocessing.py`
  - `sta_lta_detection.py`
  - `beamforming.py`
  - `fk_analysis.py`
  - `catalogue_matching.py`
  - `feature_extraction.py`
  - `classification.py`
- `quick_test/`
  - `example_data/`
  - `run_quick_test.py`
  - `README.md`
- `examples/`
  - `example_workflow.py`
- `configs/`
  - `example_config.yaml`
- `docs/`
  - `workflow.md`
  - `parameters.md`
  - `data_format.md`

> **Note:** The exact filenames in this structure should be updated to match the final source files actually uploaded to the repository.

---

## 15. Software Requirements

The workflow is implemented in Python.

The required Python packages include:

- Python
- ObsPy
- NumPy
- SciPy
- pandas
- scikit-learn
- XGBoost
- Matplotlib

The exact package versions used for the final computational environment are specified in:

`requirements.txt`

### 15.1 Recommended Environment

A dedicated Python virtual environment or Conda environment is recommended.

Example:

`python -m venv venv`

Activate the environment.

**macOS/Linux**

`source venv/bin/activate`

**Windows**

`venv\Scriptsctivate`

---

## 16. Installation

Clone the repository:

`git clone [REPOSITORY_URL]`

Move into the repository directory:

`cd array-based-dataset-construction-merapi`

Create and activate a Python environment:

`python -m venv venv`

**macOS/Linux**

`source venv/bin/activate`

Install the required dependencies:

`pip install -r requirements.txt`

> **Note:** The commands above should be verified against the final repository structure and operating-system requirements before publication.

---

## 17. Quick Test

A small example dataset is provided in:

`quick_test/example_data/`

The quick test is designed to verify that the computational environment and core workflow can be executed without requiring access to the original seismic recordings.

Run:

`python quick_test/run_quick_test.py`

The quick test should demonstrate the basic execution of the workflow and produce example output in:

`quick_test/output/`

The quick test does **not** reproduce the complete 275-event research dataset.

Instead, it provides a lightweight example for verifying that the computational implementation can be installed and executed.

Detailed quick-test instructions are provided in:

`quick_test/README.md`

---

## 18. Input Data

### 18.1 Original Research Data

The original continuous seismic waveform recordings used in the study were obtained from the UGM remote seismic array at Mount Merapi.

These original waveform data are not included in this public repository because they are subject to data-access restrictions.

The study uses five stations:

- RE5DE
- R6940
- R265F
- R7D17
- R0279

The original recordings are stored in MiniSEED format.

### 18.2 Reference Catalogues

The event-labelling process uses:

- BMKG regional earthquake catalogue
- BPPTKG volcanic event catalogue

The catalogue data used in the study are referenced in the manuscript.

### 18.3 Example Data

A synthetic or example dataset is provided for the quick-test workflow.

The example dataset is intended only to verify software execution and demonstrate the required input structure.

It should not be interpreted as the original research dataset.

---

## 19. Expected Outputs

Depending on the selected workflow, the code can generate outputs associated with:

- detected candidate events;
- validated event information;
- array-processing results;
- back-azimuth estimates;
- slowness estimates;
- beamforming results;
- extracted temporal features;
- extracted spectral features;
- machine-learning predictions;
- classification metrics;
- confusion matrices;
- ROC curves;
- AUC results; and
- learning curves.

The exact output filenames and directory structure are described in the documentation accompanying each source module.

---

## 20. Reproducibility

The repository is provided to support transparency and reproducibility of the computational workflow described in the manuscript.

The complete research workflow consists of:

1. Raw/continuous waveform recordings
2. Preprocessing
3. STA/LTA candidate detection
4. Beamforming
5. FK analysis
6. Catalogue matching
7. Manual quality control
8. Validated seismic events
9. Temporal and spectral feature extraction
10. Train/test splitting
11. SMOTE and preprocessing
12. SVM-RBF / XGBoost
13. Evaluation

Because the original seismic recordings are not publicly redistributed in this repository, the complete numerical results reported in the manuscript cannot necessarily be regenerated from the repository alone.

The repository therefore provides:

1. the computational source code;
2. installation requirements;
3. workflow documentation;
4. parameter documentation;
5. an example input;
6. a quick-test procedure; and
7. example outputs where applicable.

The limitations caused by restricted access to the original waveform data are explicitly documented.

---

## 21. Reproducing the Main Study

To reproduce the complete analysis, users require access to the original seismic waveform recordings and the corresponding reference catalogue information.

After obtaining the required data, the workflow should be executed according to the sequence described in the documentation:

1. Prepare continuous waveform data
2. Apply waveform preprocessing
3. Run STA/LTA detection
4. Evaluate candidate events using beamforming
5. Perform FK analysis
6. Match candidates with BMKG and BPPTKG catalogues
7. Perform manual quality control
8. Extract temporal and spectral features
9. Construct the labelled dataset
10. Perform stratified train–test splitting
11. Train SVM-RBF and XGBoost
12. Evaluate classification performance
13. Generate confusion matrices and ROC/AUC results

---

## 22. Data Availability

The original seismic waveform data used in this study are not included in this repository because access to the recordings is subject to data-access restrictions.

The repository therefore does not redistribute the original continuous waveform recordings.

The public repository provides example/synthetic data for testing the computational implementation.

Reference catalogue sources are described in the manuscript and should be accessed through their respective data providers where permitted.

---

## 23. Code Availability

The source code developed for the array-based seismic dataset construction and machine-learning workflow is openly available in this public repository.

The repository contains:

- source code;
- installation requirements;
- workflow documentation;
- parameter documentation;
- quick-test/example files; and
- instructions for executing the example workflow.

Repository:

**[REPOSITORY_URL]**

The repository is intended to support transparent inspection and reuse of the computational workflow described in the associated publication.

---

## 24. Citation

If you use this code or adapt the workflow in another study, please cite the associated publication:

> Dairoh, Sudarmaji, Ahmad Ashari, & Wiwit Suryanto.  
> *Array-Based Dataset Construction for Machine Learning Classification of Long-Distance Volcanic Seismic Signals.*  
> Computers & Geosciences.

DOI:

**[DOI_TO_BE_ADDED_AFTER_PUBLICATION]**

---

## 25. Authors

### Dairoh

Department of Physics, Faculty of Mathematics and Natural Sciences, Universitas Gadjah Mada, Indonesia

Informatics Engineering Study Program, Vocational School, Harkat Negeri University, Indonesia

### Sudarmaji

Department of Physics, Faculty of Mathematics and Natural Sciences, Universitas Gadjah Mada, Indonesia

### Ahmad Ashari

Department of Computer Science and Electronics, Faculty of Mathematics and Natural Sciences, Universitas Gadjah Mada, Indonesia

### Wiwit Suryanto

Department of Physics, Faculty of Mathematics and Natural Sciences, Universitas Gadjah Mada, Indonesia

---

## 26. Acknowledgement

The authors acknowledge the Universitas Gadjah Mada seismic monitoring facilities and the institutions providing reference catalogue information used for event validation.

---

## 27. License

The source code in this repository is distributed under the **MIT License**.

Please see the `LICENSE` file in the root directory of this repository for the complete license terms.

---

## 28. Disclaimer

This repository is provided for research and educational purposes.

The example or synthetic data included in the repository are intended to demonstrate software execution and do not represent the complete original seismic dataset used in the study.

The authors do not guarantee that the software will perform identically on seismic data acquired from different instruments, array geometries, volcanic environments, or observation periods without appropriate adaptation and validation.

---

## 29. Contact

For questions regarding the computational workflow or repository, please contact:

**Ahmad Ashari**

Department of Computer Science and Electronics

Faculty of Mathematics and Natural Sciences

Universitas Gadjah Mada

Yogyakarta, Indonesia

Email:

**ashari@ugm.ac.id**
