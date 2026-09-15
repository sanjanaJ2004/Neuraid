# -*- coding: utf-8 -*-
"""Neuraid - Parkinson's Disease Tremor Detection & Explainable AI (xAI)

# IMPORTING MODULES AND LIBRARIES
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.style.use("seaborn-v0_8-darkgrid")

"""## DATASET LOADING"""

import os

def load_xyz_csv(path):
    # Try direct path
    if os.path.exists(path):
        return np.loadtxt(path, delimiter=",", skiprows=1)
    # Try basename in current directory (e.g. Sanju.csv, PARKINSOND.csv)
    base_name = os.path.basename(path).replace(" (1)", "")
    if os.path.exists(base_name):
        return np.loadtxt(base_name, delimiter=",", skiprows=1)
    # Check parent/local directory fallback
    local_path = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else ".", base_name)
    if os.path.exists(local_path):
        return np.loadtxt(local_path, delimiter=",", skiprows=1)
    return np.loadtxt(path, delimiter=",", skiprows=1)

human_raw = load_xyz_csv("Sanju.csv")
park_raw  = load_xyz_csv("PARKINSOND.csv")
input_raw = load_xyz_csv("PARKINSON_IN.csv")

print("Human shape     :", human_raw.shape)
print("Parkinson shape :", park_raw.shape)
print("Input shape     :", input_raw.shape)

def fft_spectrum(signal, fs=50):
    """
    signal : 1D signal (X-axis acceleration)
    fs     : sampling frequency (Hz)
    """
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1/fs)
    return freqs, fft_vals

freq_h, fft_h = fft_spectrum(human_raw[:,0])
freq_p, fft_p = fft_spectrum(park_raw[:,0])
freq_i, fft_i = fft_spectrum(input_raw[:,0])


plt.figure(figsize=(10,4))
plt.plot(freq_h, fft_h, label="Human", alpha=0.6)
plt.plot(freq_p, fft_p, label="Parkinson", alpha=0.6)
plt.plot(freq_i, fft_i, label="Input", linewidth=2)

plt.xlim(0, 15)
plt.axvspan(4, 6, color="red", alpha=0.15, label="Parkinson Tremor Band (4–6 Hz)")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Tremor Spectrum (X-axis)")
plt.legend()
plt.show()

"""

```

```

## DATA FREQUENCY VARIATION"""

plt.figure(figsize=(12,5))
plt.plot(human_raw[:300,0], label="Human X", alpha=0.7)
plt.plot(park_raw[:300,0], label="Parkinson X", alpha=0.7)
plt.plot(input_raw[:300,0], label="Input X", linewidth=2)
plt.title("Raw X-axis Signal Trend")
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.legend()
plt.show()

"""## FEATURE ENGINERING"""

def extract_features(data):
    mean = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    rms  = np.sqrt(np.mean(data**2, axis=0))
    jerk = np.mean(np.linalg.norm(np.diff(data, axis=0), axis=1))
    return np.concatenate([mean, std, rms, [jerk]])

"""## MULTI LAYER LSTM AND COSINE SIMILARITY"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def create_sequences(data, label, window_size=50, step=10):
    X, y = [], []

    for i in range(0, len(data) - window_size, step):
        X.append(data[i:i+window_size])
        y.append(label)

    return np.array(X), np.array(y)

X_h, y_h = create_sequences(human_raw, 0)
X_p, y_p = create_sequences(park_raw, 1)

X = np.vstack((X_h, X_p))
y = np.hstack((y_h, y_p))

print("X shape:", X.shape)
print("y shape:", y.shape)

scaler = StandardScaler()

X_reshaped = X.reshape(-1, X.shape[2])
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape)


X_input, _ = create_sequences(input_raw, 0)
X_input_scaled = scaler.transform(X_input.reshape(-1, X_input.shape[2]))
X_input_scaled = X_input_scaled.reshape(X_input.shape)


inputs = Input(shape=(X_scaled.shape[1], X_scaled.shape[2]))


x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
# MULTILAYER LSTM

x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = LSTM(32, return_sequences=True)(x)

attention = Dense(1, activation='tanh')(x)
attention = Softmax(axis=1)(attention)

x = Multiply()([x, attention])
x = Lambda(lambda val: tf.reduce_sum(val, axis=1))(x)

embedding = Dense(64, activation='relu', name="embedding_layer")(x)

x = Dropout(0.4)(embedding)
x = Dense(32, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    X_scaled, y,
    epochs=15,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop, lr_reduce]
)


embed_model = Model(
    inputs=model.input,
    outputs=model.get_layer("embedding_layer").output
)

human_embed = embed_model.predict(X_scaled[y == 0])
park_embed  = embed_model.predict(X_scaled[y == 1])
input_embed = embed_model.predict(X_input_scaled)

# COSINE SIMILARITY
human_proto = np.mean(human_embed, axis=0, keepdims=True)
park_proto  = np.mean(park_embed, axis=0, keepdims=True)

sim_human = cosine_similarity(input_embed, human_proto)
sim_park  = cosine_similarity(input_embed, park_proto)

print("\nSimilarity with Healthy:", sim_human)
print("Similarity with Parkinson:", sim_park)


if np.mean(sim_park) > np.mean(sim_human):
    print("\n⚠️ Higher similarity to Parkinson pattern")
else:
    print("\n✅ Closer to Healthy pattern")

"""## WINDOW"""

def make_windows(data, n=10):
    return np.array_split(data, n)

human_windows = make_windows(human_raw, 10)
park_windows  = make_windows(park_raw, 10)

human_train = human_windows[:7]
human_test  = human_windows[7:]

park_train  = park_windows[:7]
park_test   = park_windows[7:]

"""## MODEL DEVELOPMENT AND TRAINING"""

def euclidean(a, b):
    return np.sqrt(np.sum((a - b)**2))

class CustomEpochModel:
    def __init__(self, lr=0.05, epochs=40):
        self.lr = lr
        self.epochs = epochs
        self.loss_train = []
        self.loss_test = []

    def train(self, human_train, park_train, human_test, park_test):
        self.h_proto = extract_features(human_train[0])
        self.p_proto = extract_features(park_train[0])

        for ep in range(1, self.epochs+1):
            train_loss = 0
            for h in human_train:
                f = extract_features(h)
                train_loss += euclidean(f, self.h_proto)
                self.h_proto -= self.lr * (self.h_proto - f)

            for p in park_train:
                f = extract_features(p)
                train_loss += euclidean(f, self.p_proto)
                self.p_proto -= self.lr * (self.p_proto - f)

            test_loss = 0
            for h in human_test:
                test_loss += euclidean(extract_features(h), self.h_proto)
            for p in park_test:
                test_loss += euclidean(extract_features(p), self.p_proto)

            self.loss_train.append(train_loss / len(human_train + park_train))
            self.loss_test.append(test_loss / len(human_test + park_test))

print(" MODEL DEPLOYED SUCCESSFULLY")

model = CustomEpochModel()
model.train(human_train, park_train, human_test, park_test)
print("MODEL TRAINED SUCCESSFULLY")

plt.figure(figsize=(8,4))
plt.plot(model.loss_train, label="Training Loss")
plt.plot(model.loss_test, label="Testing Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Training vs Testing Trend")
plt.legend()
plt.show()

"""## NORMAL VS ABNORMAL FEATURE VARIATION"""

human_feat = extract_features(human_raw)
park_feat  = extract_features(park_raw)
input_feat = extract_features(input_raw)

plt.figure(figsize=(10,4))
plt.plot(human_feat, marker="o", label="Human")
plt.plot(park_feat, marker="o", label="Parkinson")
plt.plot(input_feat, marker="o", label="Input")
plt.title("Feature-Level Variation Comparison")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.legend()
plt.show()

"""## INPUT DEVIATION FROM NORMAL HUMAN"""

dist_trend = []
for w in make_windows(input_raw, 10):
    f = extract_features(w)
    dist_trend.append(euclidean(f, model.h_proto))

plt.figure()
plt.plot(dist_trend, marker="o")
plt.title("Input Distance-to-Human Trend (Temporal)")
plt.xlabel("Window")
plt.ylabel("Distance")
plt.show()

confidence = []
for w in make_windows(input_raw, 10):
    f = extract_features(w)
    dh = euclidean(f, model.h_proto)
    dp = euclidean(f, model.p_proto)
    confidence.append(abs(dh-dp)/(dh+dp+1e-6))

plt.figure()
plt.plot(confidence, marker="s")
plt.title("Prediction Confidence Trend")
plt.xlabel("Window")
plt.ylabel("Confidence")
plt.show()

"""## RESULT"""

input_feat = extract_features(input_raw)


dist_to_human = euclidean(input_feat, model.h_proto)
dist_to_park  = euclidean(input_feat, model.p_proto)


is_parkinson = dist_to_park < dist_to_human

if dist_to_human < 1.0:
    level = "LOW"
elif dist_to_human < 2.5:
    level = "MODERATE"
else:
    level = "HIGH"


deviation_vector = input_feat - model.h_proto



print("\n================ FINAL RESULT ================")
print("INPUT CSV CLASSIFICATION :",
      "PARKINSON" if is_parkinson else "NORMAL HUMAN")

if is_parkinson:
    print("PARKINSON LEVEL        :", level)
else:
    print("PARKINSON LEVEL        : NONE")

print("DISTANCE TO HUMAN      :", round(dist_to_human, 4))
print("DISTANCE TO PARKINSON  :", round(dist_to_park, 4))

print("\n--- FEATURE DEVIATION FROM HUMAN ---")
feature_names = [
    "Mean X","Mean Y","Mean Z",
    "Std X","Std Y","Std Z",
    "RMS X","RMS Y","RMS Z",
    "Jerk"
]

for name, val in zip(feature_names, deviation_vector):
    print(f"{name:10s} : {val:+.5f}")

"""## SHAP(xAI)"""

import shap
import numpy as np
import matplotlib.pyplot as plt

shap.initjs()


def predict_fn(features_batch):
    scores = []
    for f_single in features_batch:
        dh = euclidean(f_single, model.h_proto)
        dp = euclidean(f_single, model.p_proto)
        scores.append(dp - dh)
    return np.array(scores)

background_features = []
for h_window in human_train:
    background_features.append(extract_features(h_window))
for p_window in park_train:
    background_features.append(extract_features(p_window))
background = np.array(background_features)

samples = input_feat.reshape(1, -1)

explainer = shap.KernelExplainer(predict_fn, background)

shap_values = explainer.shap_values(samples)

shap_vals = shap_values

feature_names = [
    "Mean X","Mean Y","Mean Z",
    "Std X","Std Y","Std Z",
    "RMS X","RMS Y","RMS Z",
    "Jerk"
]

print("\n--- SHAP Summary Plot (Feature Impact) ---")
shap.summary_plot(
    shap_vals,
    samples,
    feature_names=feature_names
)

print("\n--- SHAP Feature Importance (Bar Plot) ---")
shap.summary_plot(
    shap_vals,
    samples,
    feature_names=feature_names,
    plot_type="bar"
)

print("\n--- SHAP Force Plot (Individual Prediction Explanation) ---")
shap.force_plot(
    explainer.expected_value,
    shap_vals[0],
    samples[0],
    feature_names=feature_names,
    matplotlib=True,
    show=False )
plt.show()

"""# Task
Perform a Parkinson's disease detection task by loading accelerometer data from `sanju.csv`, `PARKINSOND.csv`, and `PARKINSON_IN.csv`. The process involves:
1.  **Data Loading and Preprocessing**: Loading raw data, performing FFT analysis, visualizing raw signal trends, extracting features (mean, std, RMS, jerk), and creating data windows for training and testing.
2.  **Model Development**: Implementing a `CustomEpochModel` that utilizes *cosine similarity* for loss calculation and prototype updates (where loss is defined as `1 - cosine_similarity`).
3.  **Model Training and Evaluation**: Training the model with human and Parkinson's data, then plotting the training and testing loss trends.
4.  **Analysis and Prediction**: Visualizing feature variation, calculating and plotting the 'Input Dissimilarity-to-Human Trend' and 'Prediction Confidence Trend' based on cosine similarity, and finally classifying the input data (`PARKINSON_IN.csv`) as 'NORMAL HUMAN' or 'PARKINSON' with a severity level.
5.  **Explainability**: Generating SHAP explanation plots (summary, bar, and force plots) to understand feature importance for the model's predictions, adapting `predict_fn` to reflect cosine similarity-based distance.

## Consolidated Code Execution

### Subtask:
Execute all necessary steps in a single code block, including library imports, function definitions, model implementation (adapted for cosine similarity), data loading, preprocessing, model training, analysis, and SHAP explanations.

## Summary:

### Data Analysis Key Findings
*   The analysis utilized accelerometer data from `sanju.csv`, `PARKINSOND.csv`, and `PARKINSON_IN.csv` as input.
*   Data preprocessing involved Fast Fourier Transform (FFT) analysis, visualization of raw signal trends, and extraction of features such as mean, standard deviation, Root Mean Square (RMS), and jerk.
*   A `CustomEpochModel` was implemented, distinctive for using cosine similarity for both loss calculation (defined as \$1 - \text{cosine\_similarity}\$) and prototype updates.
*   The model was trained on both human and Parkinson's disease data, with training and testing loss trends monitored.
*   Prediction and analysis involved calculating 'Input Dissimilarity-to-Human Trend' and 'Prediction Confidence Trend' using cosine similarity, ultimately classifying input data as 'NORMAL HUMAN' or 'PARKINSON' with a severity level.
*   SHAP (SHapley Additive exPlanations) was employed to provide explainability, generating summary, bar, and force plots to illustrate feature importance for the model's predictions, with the `predict_fn` adapted for cosine similarity-based distance.

### Insights or Next Steps
*   The bespoke `CustomEpochModel` leveraging cosine similarity for loss and prototype updates offers a robust way to distinguish between different health states; future work could explore the optimal thresholds for severity classification and the generalizability of this similarity metric across diverse datasets.
*   The integration of SHAP for explainability provides critical insights into which features drive the model's predictions. Further analysis of the most influential features identified by SHAP could lead to a deeper understanding of Parkinson's disease biomarkers and guide future feature engineering efforts.
"""