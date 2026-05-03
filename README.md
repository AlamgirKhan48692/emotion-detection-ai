# 😄 Emotion Detection AI using CNN (TensorFlow + Streamlit)

## Live Demo
https://emotion-detection-ai-alamgir.streamlit.app

An AI-powered web app that detects human emotions from facial images using a Convolutional Neural Network (CNN).

Built with **TensorFlow/Keras**, **OpenCV**, and **Streamlit**, this project allows users to upload an image and instantly receive an emotion prediction along with confidence scores.

---

## 🚀 Features

* Upload an image and detect facial emotion
* Supports multiple emotions:

  * Angry 😠
  * Disgust 🤢
  * Fear 😨
  * Happy 😄
  * Sad 😢
  * Surprise 😲
  * Neutral 😐
* Displays prediction with emoji
* Shows probability distribution (bar chart)
* Simple and interactive web interface

---

## 🧠 How It Works

The model:

* Takes a face image as input
* Converts it to grayscale
* Resizes to **48x48 pixels**
* Passes it through a trained CNN model (`emotion_cnn_final.keras`)
* Outputs probabilities for each emotion class

---

## 🖥️ Demo (How it looks)

## 📸 Screenshots

### 🏠 Home Page
![Home](screenshots/home.png)

### 📊 Prediction Result
![Result](screenshots/result.png)

---

## 📂 Project Structure

```
├── app.py                      # Streamlit app
├── emotion_cnn_final.keras     # Trained CNN model
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── LICENSE
└── .gitignore
```

---

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/AlamgirKhan48692/emotion-detection-ai.git

```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📦 Requirements

* Python 3.x
* Streamlit
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib

---

## 📊 Output Example

* Predicted Emotion → 😄 Happy
* Confidence Graph → Bar chart of all emotions

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* Streamlit
* Matplotlib

---

## 📌 Notes

* The model works best with **clear face images**
* Input image is automatically preprocessed inside the app 
* This is a **single-face prediction system** (no face detection yet)

---

## 🚧 Future Improvements

* Add real-time webcam detection
* Integrate face detection (Haar Cascade / MTCNN)
* Improve model accuracy with larger dataset
* Deploy online (Streamlit Cloud / Hugging Face)

---

## 👨‍💻 Author

**Alamgir Khan**  
📘 GitHub: https://github.com/AlamgirKhan48692  
🌐 Live App: https://emotion-detection-ai-alamgir.streamlit.app

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!

