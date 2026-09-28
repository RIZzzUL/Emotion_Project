import librosa
import numpy as np

def get_heuristics(file_path):
    y, sr = librosa.load(file_path, duration=3)
    
    # Speed (Tempo)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo[0] if isinstance(tempo, np.ndarray) else tempo
    
    # Tone / Energy
    rms = librosa.feature.rms(y=y)
    energy = np.mean(rms)
    
    # Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    # Get the pitch with max magnitude for each frame
    pitch_vals = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_vals.append(pitch)
    median_pitch = np.median(pitch_vals) if len(pitch_vals) > 0 else 0
    
    print(f"Tempo: {tempo}, Energy: {energy}, Pitch: {median_pitch}")
    
get_heuristics("temp_inference.wav")
