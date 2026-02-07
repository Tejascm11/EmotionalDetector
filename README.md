# EmotionalDetector

A real-time facial emotion detection system built using deep learning and computer vision.
The application captures live video from a webcam, detects human faces, and classifies
emotions in real time.

---

## 🚀 Features
- Real-time webcam-based emotion detection
- Supports multiple faces in a single frame
- Emotions detected:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- Derived emotional states such as:
  - Stressed
  - Excited
  - Low Mood
- Lightweight and efficient OpenCV-based implementation

---

## 🛠️ Technologies Used
- Python
- OpenCV
- TensorFlow / Keras
- NumPy

---

## ⚙️ How It Works
1. The webcam captures live video frames
2. Faces are detected using Haar Cascade classifiers
3. Each detected face is resized and normalized
4. A trained CNN model predicts the facial emotion
5. Higher-level emotions are derived using rule-based logic
6. Results are displayed in real time on the video feed

---

## 📦 Installation

Install the required dependencies:

```bash
pip install opencv-python numpy tensorflow
*Run this code by*
python webcam_emotion.py
