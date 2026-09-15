# Parkinson's Disease Tremor Detection & Explainable AI (xAI)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-yellow.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-xAI-red.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Deep Learning pipeline designed for non-invasive, objective detection of **Parkinson's Disease (PD)** tremors from tri-axial accelerometer streams. The system integrates **spectral frequency analysis (FFT)**, **kinematic feature engineering**, a **1D-CNN + Multi-Layer Bidirectional LSTM with Attention**, **Prototype Metric Learning**, and **SHAP (xAI)** for clinical interpretability.

---

## Project Overview

Parkinson's disease presents characteristic motor symptoms, most notably resting tremors exhibiting rhythmic oscillations between **4–6 Hz**. This project develops an automated diagnostics and severity assessment framework using inertial sensor data:

1. **Signal Processing & Spectral Analysis**: Isolates oscillatory frequencies using Fast Fourier Transform (FFT) with peak detection in the 4–6 Hz pathological tremor window.
2. **Kinematic Feature Extraction**: Quantifies 3D movement dynamics (Mean, Standard Deviation, Root Mean Square, and Jerk).
3. **Deep Learning Architecture**: A hybrid 1D-CNN + Multi-Layer Bi-LSTM network with a temporal self-attention mechanism to capture temporal dependencies in continuous sensor readings.
4. **Prototype Learning & Cosine Similarity**: Evaluates latent embedding proximity against healthy and pathological prototypes to compute dissimilarity trajectories and prediction confidence.
5. **Explainable AI (xAI)**: Integrates SHAP (Shapley Additive Explanations) to interpret model decisions, quantifying feature impact (e.g., Jerk vs. RMS).

---

## Architecture Pipeline

```
  +--------------------------------------------------------+
  |    Tri-Axial Accelerometer Signals (X, Y, Z Data)      |
  +--------------------------------------------------------+
                             |
         +-------------------+-------------------+
         |                                       |
         v                                       v
+-----------------------+             +-----------------------+
| FFT Spectral Analysis |             |  Kinematic Features   |
| (4-6 Hz Tremor Band)  |             | (Mean, Std, RMS, Jerk)|
+-----------------------+             +-----------------------+
         |                                       |
         +-------------------+-------------------+
                             |
                             v
           +-----------------------------------+
           |    1D-CNN Feature Extractor       |
           +-----------------------------------+
                             |
           +-----------------------------------+
           | Multi-Layer Bidirectional LSTM    |
           |     (128 -> 64 -> 32 Units)       |
           +-----------------------------------+
                             |
           +-----------------------------------+
           |    Temporal Attention Layer       |
           +-----------------------------------+
                             |
           +-----------------------------------+
           |   64-Dim Latent Embeddings        |
           +-----------------------------------+
                             |
         +-------------------+-------------------+
         |                                       |
         v                                       v
+-----------------------+             +-----------------------+
|  Cosine / Prototype   |             |     SHAP (xAI)        |
| Metric Classification |             |  Explainability Plots |
+-----------------------+             +-----------------------+
```

---

## Repository Structure

```
├── batch_20.py            # Complete end-to-end training, analysis & SHAP evaluation
├── Sanju.csv              # Healthy control accelerometer readings (X, Y, Z)
├── PARKINSOND.csv         # Parkinson's patient baseline accelerometer data
├── PARKINSON_IN.csv       # Test subject input accelerometer sample
├── requirements.txt       # Python package dependencies
├── .gitignore             # Git ignore configuration
└── README.md              # Project documentation
```

---

## Datasets

The repository includes real tri-axial sensor recordings sampled at 50 Hz:
- **`Sanju.csv`**: Baseline physiological movement data from a healthy subject.
- **`PARKINSOND.csv`**: Pathological accelerometer data from a confirmed Parkinson's patient exhibiting resting tremor.
- **`PARKINSON_IN.csv`**: Test input evaluated for diagnostic classification and severity score estimation.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sanjanaJ2004/Neuraid.git
cd Neuraid
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline
```bash
python batch_20.py
```

---

## Key Methodologies & Techniques

### 1. Spectral Analysis (FFT)
- Converts time-series sensor data to the frequency domain via Real FFT (`np.fft.rfft`).
- Visualizes tremor frequency density and highlights the distinct 4–6 Hz frequency spike characteristic of Parkinson's disease.

### 2. Feature Extraction
Computes statistical and kinematic indicators across 10 temporal windows:
- **Mean & Standard Deviation ($\mu, \sigma$)**: Static and dynamic acceleration components.
- **Root Mean Square (RMS)**: Total energy and magnitude of motion.
- **Jerk**: Rate of change of acceleration ($\Delta a / \Delta t$), measuring movement abruptness.

### 3. Deep Learning with Attention
- **Input**: Windowed 3D accelerometer sequences.
- **Encoder**: 1D Convolution with Batch Normalization and Dropout for noise suppression.
- **Recurrent Layers**: 2-layer Bidirectional LSTM followed by an LSTM layer.
- **Attention**: Dense layer with Tanh activation and Softmax normalization to dynamically weigh critical tremor frames.
- **Embeddings**: Extracted from a 64-unit bottleneck layer for metric similarity evaluation.

### 4. Model Explainability with SHAP
- Uses `shap.KernelExplainer` to compute marginal feature contributions.
- Generates:
  - **Summary Plots**: Global distribution of feature impact.
  - **Feature Importance Bars**: Ranked ranking of predictive drivers (e.g., Jerk vs. Z-axis RMS).
  - **Force Plots**: Local instance-level attribution for individual diagnostic decisions.

## License
This project is open-source under the [MIT License](LICENSE).
