import librosa
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
