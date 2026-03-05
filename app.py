
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load Model
model = load_model("emotion_cnn_final.keras")

# Emotion labels
emotions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Emojis
emoji = {
"Angry":"😠",
"Disgust":"🤢",
"Fear":"😨",
"Happy":"😄",
"Sad":"😢",
"Surprise":"😲",
"Neutral":"😐"
}

# Streamlit UI
st.title("Emotion Detection AI")
st.write("Upload a face image and the AI will detect the emotion.")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

# If Image Uploaded
if file is not None:

    # Convert uploaded file to image
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Show image
    st.image(img, channels="BGR", caption="Uploaded Image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to model input size
    face = cv2.resize(gray, (48,48))

    # Normalize
    face = face / 255.0

    # Reshape for CNN
    face = np.reshape(face, (1,48,48,1))

    # Prediction
    prediction = model.predict(face)
    prediction = prediction[0]

    emotion_index = np.argmax(prediction)
    emotion = emotions[emotion_index]

    st.subheader("Prediction Result")
    st.write(f"{emoji[emotion]} **{emotion}**")

    # Probability Chart
    st.subheader("Emotion Probability")

    fig, ax = plt.subplots()
    ax.bar(emotions, prediction)
    ax.set_ylabel("Probability")
    ax.set_title("Emotion Confidence")

    plt.xticks(rotation=45)

    st.pyplot(fig)