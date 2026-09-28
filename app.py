import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
# pyrefly: ignore [missing-import]
import librosa
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
# pyrefly: ignore [missing-import]
from tensorflow.keras.models import model_from_json
import uuid
import sqlite3
import speech_recognition as sr_audio

def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript TEXT,
            emotion TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_prediction(transcript, emotion, confidence):
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('INSERT INTO history (transcript, emotion, confidence) VALUES (?, ?, ?)', (transcript, emotion, confidence))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

def transcribe_audio(file_path):
    recognizer = sr_audio.Recognizer()
    try:
        with sr_audio.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text.lower()
    except Exception as e:
        print(f"Transcription failed: {e}")
        return ""

# ----------------------------------------------------------------------
#  Constants
# ----------------------------------------------------------------------
MODEL_ARCH    = "model.json"
MODEL_WEIGHTS = "model_weights.h5"
LABELS_FILE   = "emotion_labels.npy"

# ----------------------------------------------------------------------
#  Model / Labels loader
# ----------------------------------------------------------------------
def load_model():
    """Load the Keras model from JSON + H5 weights."""
    with open(MODEL_ARCH, "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(MODEL_WEIGHTS)
    return model

def load_labels():
    """Return list of emotion strings."""
    return np.load(LABELS_FILE, allow_pickle=True)[0].tolist()

# ----------------------------------------------------------------------
#  Feature extraction (same as training)
# ----------------------------------------------------------------------
def extract_features(file_path):
    """
    Extracts audio features using Librosa.
    The algorithm extracts 42 features across time steps:
    - 40 Mel-Frequency Cepstral Coefficients (MFCCs)
    - 1 Zero-Crossing Rate (ZCR)
    - 1 Root Mean Square (RMS) energy
    
    It then returns a (130, 42) feature matrix suitable for 1D CNN + LSTM.
    """
    try:
        # Load the audio file, ignoring the first 0.5 seconds, and grabbing 3 seconds of audio.
        y, sr = librosa.load(file_path, duration=3, offset=0.5)

        # Ensure the audio is exactly 3 seconds long.
        target_len = sr * 3
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), "constant")
        elif len(y) > target_len:
            y = y[:target_len]

        # ── Domain Adaptation Step 1: Pre-emphasis filter ─────────────────────
        # Boosts high frequencies to compensate for laptop mic roll-off.
        # MUST match train.py exactly so inference features == training features.
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])

        # ── Domain Adaptation Step 2: RMS Normalisation ───────────────────────
        # Equalises volume differences between microphones and recording distances.
        rms_val = np.sqrt(np.mean(y ** 2))
        if rms_val > 1e-8:
            y = y / rms_val

        # 1. Extract 40 MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # 2. Extract Zero-Crossing Rate
        zcr  = librosa.feature.zero_crossing_rate(y=y)
        # 3. Extract RMS Energy
        rms  = librosa.feature.rms(y=y)

        # ── Domain Adaptation Step 3: CMVN ────────────────────────────────────
        # Removes microphone frequency-response colouration by centering each
        # MFCC coefficient to zero mean and unit variance across time frames.
        mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
        mfcc_std  = np.std(mfcc,  axis=1, keepdims=True) + 1e-8
        mfcc = (mfcc - mfcc_mean) / mfcc_std

        # Transpose and stack: (42, 130) -> (130, 42)
        features = np.vstack((mfcc, zcr, rms)).T
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def extract_acoustics(file_path):
    """
    Extracts Pitch, Tone (Energy/Brightness), and Speed.
    Returns categorical human states (e.g., is_pitch_high, is_fast) based on user definitions.
    """
    try:
        y, sr = librosa.load(file_path, duration=3)
        
        # 1. Speed (Tempo)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo[0] if isinstance(tempo, np.ndarray) else tempo
        
        # 2. Tone (Energy and Brightness)
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))
        centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        brightness = float(np.mean(centroids))
        
        # 3. Pitch (Fundamental Frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if 50 < pitch < 500:
                pitch_vals.append(pitch)
                
        median_pitch = float(np.median(pitch_vals)) if len(pitch_vals) > 0 else 0.0
        pitch_variation = float(np.std(pitch_vals)) if len(pitch_vals) > 0 else 0.0

        print(f"[DEBUG Acoustics] Tempo: {tempo:.1f}, Energy: {energy:.3f}, Pitch: {median_pitch:.1f}")

        return {
            "is_fast": tempo > 125,
            "is_medium_speed": 100 <= tempo <= 125,
            "is_pitch_high": median_pitch > 170 or pitch_variation > 40,
            "is_pitch_low": median_pitch < 140 or pitch_variation < 20,
            "is_tone_high": energy > 0.04 or brightness > 2500,
            "is_tone_medium": 0.015 <= energy <= 0.04,
            "is_tone_low": energy < 0.015
        }
            
    except Exception as e:
        print(f"Error extracting acoustics: {e}")
        return {
            "is_fast": False, "is_medium_speed": True, "is_pitch_high": False,
            "is_pitch_low": False, "is_tone_high": False, "is_tone_medium": True, "is_tone_low": False
        }

# ----------------------------------------------------------------------
#  Flask app
# ----------------------------------------------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Serve the UI at the root – this gives us same‑origin permissions.
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# Load model / labels once at startup.
# Lazy-load so the server can boot (and report status via /health)
# even while model files are being written by a training run.
model = None
labels = None
_model_error = None


def ensure_model_loaded():
    global model, labels, _model_error
    if model is not None and labels is not None:
        return True
    try:
        model = load_model()
        labels = load_labels()
        _model_error = None
        return True
    except Exception as e:
        _model_error = str(e)
        return False


@app.route("/health", methods=["GET"])
def health():
    ok = ensure_model_loaded()
    status = 200 if ok else 503
    return jsonify({
        "model_loaded": ok,
        "labels": labels if ok else [],
        "error": _model_error if not ok else None,
    }), status

@app.route("/predict", methods=["POST"])
def predict():
    if not ensure_model_loaded():
        return jsonify({"error": f"Model not ready yet: {_model_error}. Training may still be running."}), 503
    if "audio_file" not in request.files:
        return jsonify({"error": "No audio_file found in request."}), 400

    file = request.files["audio_file"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    # Save the uploaded file securely with a unique UUID to prevent race conditions 
    # when multiple users predict simultaneously.
    unique_filename = f"temp_inference_{uuid.uuid4().hex}.wav"
    tmp_path = unique_filename
    try:
        file.save(tmp_path)

        # 1. Extract Text
        transcript = transcribe_audio(tmp_path)

        # 2. Extract Acoustic Categories
        acoustics = extract_acoustics(tmp_path)

        # 3. Extract model features and run prediction
        features = extract_features(tmp_path)
        if features is None:
            raise ValueError("Could not extract audio features for model inference.")

        model_input = np.expand_dims(features.astype(np.float32), axis=0)
        probabilities = model.predict(model_input, verbose=0)[0]
        best_idx = int(np.argmax(probabilities))
        model_emotion = labels[best_idx]
        model_confidence = float(probabilities[best_idx])

        has_positive_words = False
        has_sad_words = False
        has_angry_words = False

        if transcript:
            has_positive_words = any(w in transcript for w in ["happy", "good", "great", "joy", "awesome", "positive", "birthday", "love", "nice", "happieee"])
            has_sad_words = any(w in transcript for w in ["sad", "cry", "depressed", "upset", "sorry", "bad"])
            has_angry_words = any(w in transcript for w in ["angry", "mad", "hate", "furious", "terrible"])

        final_emotion = model_emotion
        final_confidence = model_confidence

        # Confidence-based fallback: apply heuristic rules when the model is unsure.
        if final_confidence < 0.75:
            if acoustics["is_pitch_high"] and acoustics["is_tone_high"] and acoustics["is_fast"]:
                final_emotion = "Excited"
                final_confidence = 0.95
            elif acoustics["is_pitch_high"] and acoustics["is_tone_medium"] and acoustics["is_fast"]:
                final_emotion = "Angry"
                final_confidence = 0.88
            elif acoustics["is_pitch_high"] and has_positive_words:
                final_emotion = "Happy"
                final_confidence = 0.92
            elif acoustics["is_pitch_low"] and acoustics["is_tone_low"] and acoustics["is_medium_speed"]:
                final_emotion = "Sad"
                final_confidence = 0.85
            elif has_sad_words:
                final_emotion = "Sad"
                final_confidence = 0.88
            elif has_angry_words:
                final_emotion = "Angry"
                final_confidence = 0.88
            elif has_positive_words:
                final_emotion = "Happy"
                final_confidence = 0.85

        # Log to Database
        log_prediction(transcript, final_emotion, final_confidence)

        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return jsonify({"emotion": final_emotion, "confidence": final_confidence, "transcript": transcript})
    except Exception as e:
        print(f"Server Error: {e}")
        # Clean up on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🎙️  Open http://127.0.0.1:5001 in your browser.")
    print("   Microphone & speaker permissions will work over HTTP.\n")
    app.run(host="127.0.0.1", port=5001, debug=True)
