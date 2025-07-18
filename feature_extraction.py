import os
import librosa
import pandas as pd
from tqdm import tqdm
import numpy as np

def extract_features(file_path):
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # 1. MFCCs - capture tone & voice fingerprint
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)

    # 2. Chroma - pitch-related features
    stft = np.abs(librosa.stft(audio))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # 3. Mel Spectrogram - energy in different frequency bands
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)

    # Combine all extracted features into one feature vector
    combined = np.hstack([mfccs, chroma, mel])
    return combined



# Emotion label map
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

# Dataset path
data_path = "Emotion database\\audio_speech_actors_01-24"
features = []
labels = []

# Walk through dataset
for actor in tqdm(os.listdir(data_path)):
    actor_path = os.path.join(data_path, actor)
    for file in os.listdir(actor_path):
        file_path = os.path.join(actor_path, file)
        emotion_code = file.split("-")[2]  # e.g., 03-01-**03**-01...
        emotion_label = emotion_map[emotion_code]
        feat = extract_features(file_path)
        features.append(feat)
        labels.append(emotion_label)

# Convert to DataFrame
df_features = pd.DataFrame(features)
df_features['label'] = labels
df_features.to_csv("emotion_features.csv", index=False)

