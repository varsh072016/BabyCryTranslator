import os
import numpy as np
import soundfile as sf
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# =========================
# CONFIG
# =========================
DATASET_PATH = "data"  # dataset/class_name/*.wav

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(file_path):
    signal, sr = sf.read(file_path)
    if signal.ndim > 1:
        signal = signal[:, 0]

    # Basic statistics
    features = [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        skew(signal),
        kurtosis(signal)
    ]

    # RMS energy
    rms = np.sqrt(np.mean(signal**2))
    features.append(rms)

    # Zero-crossing rate
    zcr = ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)
    features.append(zcr)

    # FFT energy bands (10 bands)
    spectrum = np.abs(np.fft.rfft(signal))
    bands = np.array_split(spectrum, 10)
    features.extend([np.mean(b) for b in bands])

    return np.array(features)

# =========================
# LOAD DATASET
# =========================
X = []
y = []

for label in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):
        if file.lower().endswith(".wav"):
            file_path = os.path.join(class_path, file)
            try:
                feats = extract_features(file_path)
                X.append(feats)
                y.append(label)
            except Exception as e:
                print("Error:", file_path, e)

X = np.array(X)
y = np.array(y)
print(f"Total samples: {len(X)}")
print(f"Feature vector size: {X.shape[1]}")

# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes found:", list(le.classes_))

# =========================
# TRAIN/TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =========================
# MODEL TRAINING
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL & ENCODER
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("\n✅ Model retrained and saved successfully!")
