from flask import Flask, render_template, request
import numpy as np
import pickle
import soundfile as sf
from scipy.stats import skew, kurtosis
from scipy.fft import rfft
import os
from datetime import datetime

app = Flask(__name__)

# Load model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# Feature extraction
# =========================
def extract_features(file_path):
    signal, sr = sf.read(file_path)
    if signal.ndim > 1:
        signal = signal[:, 0]

    features = [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        skew(signal),
        kurtosis(signal),
        np.sqrt(np.mean(signal**2)),  # RMS
        ((signal[:-1]*signal[1:]) < 0).sum() / len(signal)  # ZCR
    ]

    # FFT bands
    spectrum = np.abs(rfft(signal))
    bands = np.array_split(spectrum, 10)
    features.extend([np.mean(b) for b in bands])

    # Pad if needed
    while len(features) < model.n_features_in_:
        features.append(0)

    return np.array(features).reshape(1, -1)

# =========================
# Flask routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    confidence_msg = None

    if request.method == "POST":
        audio = request.files["audio"]
        filename = f"{datetime.now().timestamp()}_{audio.filename}"
        path = os.path.join(UPLOAD_FOLDER, filename)
        audio.save(path)

        features = extract_features(path)
        probs = model.predict_proba(features)[0]

        top3_idx = np.argsort(probs)[::-1][:3]
        labels = encoder.inverse_transform(top3_idx)
        confidences = [round(probs[i]*100, 1) for i in top3_idx]
        results = list(zip(labels, confidences))

        # Friendly, cautious confidence message
        top_conf = confidences[0]
        if top_conf >= 70:
            confidence_msg = f"Our model predicts your baby may be crying due to {labels[0]}. Please observe and confirm."
        elif top_conf >= 40:
            confidence_msg = f"It looks like your baby might be crying because of {labels[0]}, but other possibilities exist."
        else:
            top_labels = ", ".join(labels)
            confidence_msg = f"The reason for crying is unclear. Top guesses: {top_labels}. Monitor and see which fits best."

    return render_template("index.html", results=results, confidence_msg=confidence_msg)

# =========================
# Run Flask
# =========================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
