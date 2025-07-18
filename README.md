# 🎙️ Human Emotion Detection from Voice

This project is a Python-based emotion detection system that uses machine learning and audio processing to identify human emotions from voice recordings. It features a modern **Streamlit UI**, voice recording support, real-time emotion prediction, and optional Firebase integration for saving audio data.

---

## 🔍 Features

- 🎤 Record audio from the browser
- 🔊 Extract audio features (MFCC, Chroma, Mel)
- 🧠 Predict emotions using a trained ML model (Random Forest/SVM)
- 📈 View emotion trend graph
- ☁️ Optional Firebase integration for storing voice data
- 💻 Web app built with Streamlit

---

## 📂 Project Structure

emotion-detection-voice/
├── app.py # Main Streamlit app
├── model.pkl # Trained machine learning model
├── record_audio.py # Handles audio recording
├── feature_extraction.py # Audio feature extraction logic
├── firebase_key.json # Firebase credentials (not uploaded)
├── requirements.txt
├── README.md
└── audio_files/ # Folder to store recorded audio

yaml
Copy
Edit

---

## 🚀 How to Run Locally

### 1. Clone the repo

"""bash
git clone https://github.com/HaswinAI/emotion-detection-voice.git
cd emotion-detection-voice

2. Create virtual environment

python -m venv audio_libs
audio_libs\Scripts\activate   # On Windows

3. Install dependencies

pip install -r requirements.txt

4. Add Firebase Key (optional)
Go to Firebase Console → Project Settings → Service Accounts

Click "Generate new private key" and download the JSON file

Save it as firebase_key.json in the project folder

Update app.py with your database URL

5. Run the Streamlit App

streamlit run app.py
'''
🧠 Supported Emotions
Neutral

Happy

Sad

Angry

Fearful

Disgust

Surprised

Trained on the RAVDESS dataset

📊 Model Performance
Class	Precision	Recall	F1-Score
Happy	0.94	0.92	0.93
Sad	0.91	0.89	0.90
Angry	0.95	0.94	0.94
...	...	...	...

Final model used: Random Forest with 93% accuracy

☁️ Firebase Integration (Optional)
Stores audio recordings with emotion labels in Firebase Realtime DB

Requires service account key (firebase_key.json)

Uses firebase-admin SDK

🔒 Security Note
Never upload your firebase_key.json or private credentials to GitHub

Add it to .gitignore:

bash
Copy
Edit
firebase_key.json
.streamlit/secrets.toml
audio_files/
🛠 Built With
Python

Streamlit

Librosa

Firebase

Scikit-learn

📸 Screenshots
Recording	Emotion Prediction

🤖 Future Enhancements
🎯 Live audio input (microphone)

📊 Emotion trends over time

📁 Multi-user login (via Firebase Auth)

🌐 Deploy on Streamlit Cloud

🧑‍💻 Author
Haswin Deepak
BTech AI-ML, Panimalar Engineering College
🔗 LinkedIn | 🌐 Portfolio | 💡 Passionate about AI & Audio Intelligence

📜 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---
