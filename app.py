import streamlit as st
import numpy as np
import librosa
import librosa.display
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib  # joblib is better for loading ML models

model_path = r"C:\Users\asus\speech_emotion_model.pkl"
scaler_path = r"C:\Users\asus\scaler.pkl"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Trained model file not found at {model_path}! Please train and save your model first.")

try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error(f"Scaler file not found at {scaler_path}! Please train and save your scaler first.")



# Streamlit UI
st.title("Speech Emotion Recognition")
st.write("Upload an audio file to analyze its emotion.")

# File Upload
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    # Process the uploaded file
    st.audio(uploaded_file, format='audio/wav')
    
    # Convert the audio file into features
    try:
        y, sr = librosa.load(uploaded_file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features = np.mean(mfccs.T, axis=0)  # Convert to a fixed-size feature vector

        # Standardize the features
        features = scaler.transform([features])

        # Make prediction
        prediction = model.predict(features)
        emotion_labels = ["Angry", "Happy", "Sad", "Neutral", "Fear", "Disgust", "Surprise"]
        
        # Display result
        st.write(f"Predicted Emotion: {emotion_labels[prediction[0]]}")
    except Exception as e:  # Catch any errors and display them
        st.error(f"An error occurred while processing the audio: {e}")
