import sys
import os
import numpy as np

# Conditional imports to prevent thread-pool and fork deadlocks in Keras/TensorFlow on macOS
if "--train-only" in sys.argv:
    # Subprocess imports (TensorFlow Keras training only)
    # pyrefly: ignore [missing-import]
    from tensorflow.keras.models import Sequential 
    # pyrefly: ignore [missing-import]
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D
    # pyrefly: ignore [missing-import]
    from tensorflow.keras.callbacks import ModelCheckpoint
else:
    # Parent process imports (Audio feature extraction only)
    import pandas as pd
    # pyrefly: ignore [missing-import]
    import librosa
    from sklearn.preprocessing import OneHotEncoder
    import subprocess


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        # Pad or truncate to exactly 3 seconds (66150 samples at sr=22050)
        target_length = sr * 3
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        elif len(y) > target_length:
            y = y[:target_length]

        # ── Domain Adaptation Step 1: Pre-emphasis filter ──────────────────────
        # Boosts high frequencies to compensate for laptop mic roll-off.
        # Formula: y[t] = y[t] - 0.97 * y[t-1]
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])

        # ── Domain Adaptation Step 2: RMS Normalisation ─────────────────────────
        # Ensures volume differences between a studio boom mic and a laptop mic
        # don't influence the emotion prediction.
        rms_val = np.sqrt(np.mean(y ** 2))
        if rms_val > 1e-8:
            y = y / rms_val

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        zcr  = librosa.feature.zero_crossing_rate(y=y)
        rms  = librosa.feature.rms(y=y)

        # ── Domain Adaptation Step 3: CMVN ──────────────────────────────────────
        # Cepstral Mean and Variance Normalisation removes the microphone's
        # "colour" (its unique frequency response) by centering each MFCC
        # coefficient to zero mean and unit variance across all time frames.
        # This is the most powerful domain-adaptation trick in speech processing.
        mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
        mfcc_std  = np.std(mfcc,  axis=1, keepdims=True) + 1e-8
        mfcc = (mfcc - mfcc_mean) / mfcc_std

        # Stack features: shape (42, 130) -> transpose to (130, 42)
        features = np.vstack((mfcc, zcr, rms)).T
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


def run_parent():
    print("--- Starting Feature Extraction Pipeline ---")

    # Both the original RAVDESS dataset AND the augmented variants are used
    DATA_PATHS = [
        "./Audio_Speech_Actors_01-24/",
        "./Augmented_Dataset/",
    ]

    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    EMOTIONS_LIST = ["angry", "sad", "happy", "neutral", "fearful"]

    train_data = []
    test_data = []

    for DATA_PATH in DATA_PATHS:
        if not os.path.isdir(DATA_PATH):
            print(f"Skipping missing folder: {DATA_PATH}")
            continue

        actor_folders = sorted([d for d in os.listdir(DATA_PATH) if d.startswith("Actor_")])

        for actor_dir in actor_folders:
            actor_num = int(actor_dir.split('_')[1])
            actor_path = os.path.join(DATA_PATH, actor_dir)
            file_list = sorted(os.listdir(actor_path))

            for file_name in file_list:
                if not file_name.endswith('.wav'):
                    continue
                # Strip augmentation suffix before parsing RAVDESS code
                base = file_name.split('.')[0]
                for suffix in ('_noisy', '_slow', '_fast', '_pitchup'):
                    base = base.replace(suffix, '')
                parts = base.split('-')
                if len(parts) < 7:
                    continue
                emotion_code = parts[2]
                emotion = emotion_map.get(emotion_code)
                if emotion in EMOTIONS_LIST:
                    file_path = os.path.join(actor_path, file_name)
                    features = extract_features(file_path)
                    if features is not None:
                        # Speaker-independent split: Actors 21-24 are test
                        # Augmented files from those actors also go to test
                        if actor_num >= 21:
                            test_data.append([features, emotion])
                        else:
                            train_data.append([features, emotion])

    print(f"--- Extracted features for {len(train_data)} train and {len(test_data)} test audio files ---")

    df_train = pd.DataFrame(train_data, columns=['features', 'emotion'])
    df_test = pd.DataFrame(test_data, columns=['features', 'emotion'])

    X_train = np.array(df_train['features'].tolist()).astype(np.float32)
    y_train_raw = np.array(df_train['emotion'].tolist())

    X_test = np.array(df_test['features'].tolist()).astype(np.float32)
    y_test_raw = np.array(df_test['emotion'].tolist())

    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train_raw.reshape(-1, 1)).astype(np.float32)
    y_test = encoder.transform(y_test_raw.reshape(-1, 1)).astype(np.float32)

    print(f"--- Saving extracted features to temp files ---")
    np.save("temp_X_train.npy", X_train)
    np.save("temp_X_test.npy", X_test)
    np.save("temp_y_train.npy", y_train)
    np.save("temp_y_test.npy", y_test)
    np.save('emotion_labels.npy', encoder.categories_)

    print("--- Starting separate training process to prevent library conflicts ---")
    script_path = os.path.abspath(__file__)
    res = subprocess.run([sys.executable, script_path, "--train-only"])
    
    # Cleanup temp files
    for temp_file in ["temp_X_train.npy", "temp_X_test.npy", "temp_y_train.npy", "temp_y_test.npy"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    if res.returncode == 0:
        print("\n--- Pipeline Completed Successfully! Model and weights saved. ---")
        print("You can now run 'python3 app.py'")
    else:
        print("\n--- Training subprocess failed! ---")
        sys.exit(res.returncode)


def run_child():
    print("--- Subprocess: Loading Features ---")
    X_train = np.load("temp_X_train.npy")
    X_test = np.load("temp_X_test.npy")
    y_train = np.load("temp_y_train.npy")
    y_test = np.load("temp_y_test.npy")

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    model = Sequential()
    # Conv1D extracts spatial acoustic patterns
    model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # LSTM models sequence temporal dependencies
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Fully connected block
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    print("--- Subprocess: Starting Model Training ---")
    checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        callbacks=[checkpoint],
        verbose=2
    )

    print("--- Subprocess: Saving Model Architecture ---")
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    
    print("--- Subprocess: Model saved successfully ---")


if __name__ == "__main__":
    if "--train-only" in sys.argv:
        run_child()
    else:
        run_parent()
