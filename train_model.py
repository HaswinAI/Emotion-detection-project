import os
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Path to the dataset (adjust to your real RAVDESS path)
DATASET_PATH = r'Emotion database\audio_speech_actors_01-24'  # or 'data/ravdess' if inside subfolder

# Emotions map (as per RAVDESS)
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

X, y = [], []

print("🔍 Extracting features...")

# Traverse dataset directory
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            emotion_code = file.split("-")[2]  # 3rd field in filename
            emotion_label = emotions.get(emotion_code)
            if emotion_label is not None:
                feature = extract_features(file_path)
                if feature is not None:
                    X.append(feature)
                    y.append(emotion_label)

print(f"✅ Feature extraction complete. Total samples: {len(X)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train classifier
print("🎯 Training model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print("✅ Model training complete!")

# Save model
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved to 'emotion_model.pkl'")
