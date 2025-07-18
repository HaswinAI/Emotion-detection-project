import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import write
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

import firebase_admin
from firebase_admin import credentials, db

# Load the model
model = joblib.load("emotion_model.pkl")
cred = credentials.Certificate("audio-sentiment-d5687-firebase-adminsdk-fbsvc-d66827d5b4.json")

# Initialize Firebase app
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://audio-sentiment-d5687-default-rtdb.firebaseio.com/'
    })

# Constants
SAMPLE_RATE = 22050
DURATION = 5  # max duration in seconds
SILENCE_THRESHOLD = 0.01  # adjust based on environment

# Emotion history
if "history" not in st.session_state:
    st.session_state.history = []

# Feature Extraction
def extract_features(audio_data, sample_rate):
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs

# Record until silence
def record_with_silence(sample_rate=SAMPLE_RATE, silence_threshold=SILENCE_THRESHOLD, max_duration=DURATION):
    st.info("Recording... Speak now!")
    recorded_audio = []
    silence_counter = 0
    frame_duration = 0.5  # seconds
    frames_per_buffer = int(sample_rate * frame_duration)

    def callback(indata, frames, time, status):
        nonlocal silence_counter
        volume = np.linalg.norm(indata)
        if volume < silence_threshold:
            silence_counter += 1
        else:
            silence_counter = 0
        recorded_audio.append(indata.copy())
        if silence_counter >= 3:  # Stop after 1.5 sec silence
            raise sd.CallbackStop()

    try:
        with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, blocksize=frames_per_buffer):
            sd.sleep(int(max_duration * 1000))
    except sd.CallbackStop:
        pass

    st.success("Recording complete.")
    audio_np = np.concatenate(recorded_audio, axis=0).flatten()
    return audio_np

def store_emotion_result(file_name, emotion):
    ref = db.reference('emotion_logs')
    ref.push({
        'filename': file_name,
        'predicted_emotion': emotion,
        'timestamp': datetime.now().isoformat()
    })

# UI
st.title("🎤 Real-Time Emotion Detector from Voice")

# Record button
if st.button("🎙️ Start Recording"):
    audio_np = record_with_silence()
    audio_path = "recorded_audio.wav"
    write(audio_path, SAMPLE_RATE, (audio_np * 32767).astype(np.int16))

    st.audio(audio_path, format='audio/wav')

    # Emotion prediction
    features = extract_features(audio_np, SAMPLE_RATE).reshape(1, -1)
    prediction = model.predict(features)[0]

    # Store and plot
    timestamp = datetime.now().strftime('%H:%M:%S')
    st.session_state.history.append((timestamp, prediction))
    #call prediction
    if st.button("Predict Emotion"):
        predicted_emotion = predict_emotion("recorded_audio.wav")
        st.success(f"Predicted Emotion: {predicted_emotion}")


    st.markdown(f"### 😃 Detected Emotion: **{prediction}**")
    # ✅ Now it's safe to log into Firebase
    store_emotion_result("recorded_audio.wav", prediction)

# Upload button
st.markdown("### 📤 Or Upload a WAV/MP3 File for Emotion Detection")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Load and preprocess uploaded file
    try:
        y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
        features = extract_features(y, sr).reshape(1, -1)
        prediction = model.predict(features)[0]

        # Play audio and show prediction
        st.audio(uploaded_file, format="audio/wav")
        timestamp = datetime.now().strftime('%H:%M:%S')
        st.session_state.history.append((timestamp, prediction))
        st.markdown(f"### 😃 Detected Emotion: **{prediction}**")
    except Exception as e:
        st.error(f"Error processing audio: {e}")

# Graph
if len(st.session_state.history) > 1:
    times, emotions = zip(*st.session_state.history)
    le = LabelEncoder()
    y_encoded = le.fit_transform(emotions)
    fig, ax = plt.subplots()
    ax.plot(times, y_encoded, marker='o', linestyle='-', color='blue')
    ax.set_yticks(range(len(le.classes_)))
    ax.set_yticklabels(le.classes_)
    ax.set_title("Emotion Trend")
    ax.set_xlabel("Time")
    ax.set_ylabel("Emotion")
    plt.xticks(rotation=45)
    st.pyplot(fig)
