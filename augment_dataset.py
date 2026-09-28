"""
Data Augmentation Script for Speech Emotion Recognition
========================================================
This script reads every .wav file in your RAVDESS dataset (Audio_Speech_Actors_01-24)
and generates 4 augmented versions of each file:
  1. Noisy     — adds random background noise (simulates a real microphone)
  2. Stretched — time-stretched to 0.8× speed  (simulates slow speech)
  3. Fast      — time-stretched to 1.2× speed  (simulates fast speech)
  4. Pitch Up  — pitch-shifted +2 semitones     (simulates a higher-pitched voice)

Output folder: Augmented_Dataset/  (mirrors the Actor_XX subfolder structure)
Labels are preserved: the RAVDESS filename already encodes emotion, so augmented
files keep the same name with a suffix (_noisy, _slow, _fast, _pitchup).

Run with:
    source venv/bin/activate
    python augment_dataset.py
"""

import os
import glob
import numpy as np # type: ignore
import librosa # type: ignore
import soundfile as sf

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_ROOT  = "Audio_Speech_Actors_01-24"
OUTPUT_ROOT = "Augmented_Dataset"
SR          = 22050          # sample rate used throughout the project

NOISE_AMPLITUDE = 0.005     # how loud the added noise is (relative to signal peak)
SLOW_RATE       = 0.8       # time-stretch ratio for "slow"
FAST_RATE       = 1.2       # time-stretch ratio for "fast"
PITCH_STEPS     = 2         # semitones to shift up for "pitch up"
# ───────────────────────────────────────────────────────────────────────────────


def add_noise(y):
    """Add Gaussian white noise – simulates a cheap or far-away microphone."""
    noise_amp = NOISE_AMPLITUDE * np.random.uniform() * np.amax(np.abs(y))
    return y + noise_amp * np.random.normal(size=y.shape[0])


def add_reverb(y, sr=SR):
    """
    Simulate simple room reverb by convolving the signal with a synthetic
    impulse response (decaying exponential). This forces the model to learn
    that the same emotion sounds different in a real room vs. a silent studio.
    """
    reverb_length = int(sr * 0.5)          # 500 ms tail
    decay = np.exp(-6 * np.linspace(0, 1, reverb_length))
    impulse_response = np.random.randn(reverb_length) * decay
    y_rev = np.convolve(y, impulse_response, mode='full')[:len(y)]
    # Normalise to avoid clipping
    peak = np.amax(np.abs(y_rev))
    if peak > 1e-8:
        y_rev = y_rev / peak * np.amax(np.abs(y))
    return y_rev.astype(np.float32)


def bandpass_filter(y, sr=SR, low_hz=300, high_hz=3400):
    """
    Apply a bandpass filter to simulate the limited frequency response of a
    laptop or webcam microphone (typically rolls off below 300 Hz and above
    ~3.4 kHz compared to a studio condenser mic).
    """
    from scipy.signal import butter, sosfilt
    sos = butter(4, [low_hz, high_hz], btype='band', fs=sr, output='sos')
    return sosfilt(sos, y).astype(np.float32)


def time_stretch(y, rate):
    """Speed the audio up or slow it down without changing pitch."""
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y, sr, n_steps):
    """Shift the pitch up or down by n_steps semitones."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def save(y, sr, path):
    """Save a numpy array as a WAV file, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr)


def augment_file(src_path):
    """
    Load one audio file and write 4 augmented variants.
    Returns the number of new files created (6) or 0 on error.
    """
    try:
        y, sr = librosa.load(src_path, sr=SR)
    except Exception as e:
        print(f"  ✗  Could not load {src_path}: {e}")
        return 0

    # Build the mirrored output sub-directory
    rel_path  = os.path.relpath(src_path, INPUT_ROOT)   # e.g.  Actor_01/03-01-01...wav
    actor_dir = os.path.dirname(rel_path)               # e.g.  Actor_01
    basename  = os.path.splitext(os.path.basename(rel_path))[0]  # without .wav
    out_dir   = os.path.join(OUTPUT_ROOT, actor_dir)

    variants = {
        "noisy":    add_noise(y),
        "slow":     time_stretch(y, SLOW_RATE),
        "bandpass": bandpass_filter(y, SR),  # Added bandpass variant
        "reverb":   add_reverb(y, SR),  # Added reverb variant
        "fast":     time_stretch(y, FAST_RATE),
        "reverb":   add_reverb(y, SR),  # Added reverb variant
        "reverb":   add_reverb(y, SR),
        "bandpass": bandpass_filter(y, SR),
    }

    for tag, y_aug in variants.items():
        out_path = os.path.join(out_dir, f"{basename}_{tag}.wav")
        save(y_aug, SR, out_path)

    return len(variants)


def main():
    wav_files = sorted(glob.glob(
        os.path.join(INPUT_ROOT, "**", "*.wav"), recursive=True
    ))

    if not wav_files:
        print(f"❌  No .wav files found under '{INPUT_ROOT}'. "
              f"Make sure you're running this script from the project root.")
        return

    total_original  = len(wav_files)
    total_augmented = 0

    print(f"\n🎙️  Speech Emotion — Data Augmentation")
    print(f"   Source  : {INPUT_ROOT}/")
    print(f"   Output  : {OUTPUT_ROOT}/")
    print(f"   Files   : {total_original} original .wav files found")
    print(f"   Goal    : {total_original * 6:,} augmented files\n")
    print("   Starting augmentation …\n")

    for i, src in enumerate(wav_files, 1):
        n = augment_file(src)
        total_augmented += n
        # Print progress every 50 files
        if i % 50 == 0 or i == total_original:
            pct = 100 * i / total_original
            print(f"   [{i:>4}/{total_original}] {pct:5.1f}%  — {total_augmented:,} files created so far")

    print(f"\n✅  Done!")
    print(f"   Original  files : {total_original:,}")
    print(f"   Augmented files : {total_augmented:,}")
    print(f"   Total in dataset: {total_original + total_augmented:,}")
    print(f"\n   Now update your training script to also read from '{OUTPUT_ROOT}/'")
    print("   and retrain your model for significantly better accuracy.\n")


if __name__ == "__main__":
    main()
