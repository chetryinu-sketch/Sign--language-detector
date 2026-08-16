# 🤟 Sign Language Detector

A Computer Vision project that detects and recognises selected American Sign Language (ASL) hand gestures using a webcam.

The current model recognises three gestures:

- 👋 Hello
- 🙏 Thank You
- ❤️ I Love You

The model was trained using **Teachable Machine** and integrated with **Python and OpenCV** for webcam-based detection.

---

## 📌 Project Overview

This project uses a trained image classification model to identify hand signs captured through a webcam.

The webcam continuously captures frames, and the trained model analyses the hand gesture and predicts the corresponding sign.

The prediction is displayed on the screen along with the confidence score.

This project was created as part of my learning journey in **Computer Vision and Machine Learning**.

---

## ✨ Features

- Real-time hand sign detection through a webcam
- Recognises selected ASL gestures
- Displays the predicted gesture
- Shows the confidence score
- Uses a custom-trained model
- Simple and beginner-friendly implementation
- Can be extended with additional hand signs

---

## 🖐️ Supported Gestures

| Gesture | Meaning |
|--------|---------|
| 👋 Hello | Greeting |
| 🙏 Thank You | Expressing gratitude |
| ❤️ I Love You | Expressing affection |

> **Note:** This project currently recognises only the three gestures listed above. It is not a complete American Sign Language translation system.

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV**
- **NumPy**
- **Teachable Machine**
- **TensorFlow/Keras Model**
- **MediaPipe / Hand Landmark Model** (if used in the implementation)
- **VS Code**

---

## 🧠 Model Training

The classification model was trained using **Google Teachable Machine**.

### Training process

1. Created separate classes for each gesture.
2. Collected multiple images for each hand sign.
3. Trained the image classification model.
4. Tested the model with different hand positions.
5. Exported the trained model.
6. Integrated the model with Python and OpenCV.

The trained model is stored in the `Model` folder.

---

## 📂 Project Structure

```text
sign-language-detector/
│
├── Data/
│   ├── Hello/
│   ├── Thank you/
│   └── I Love you/
│
├── Model/
│   ├── keras_model.h5
│   └── labels.txt
│
├── datacollection.py
├── hand_landmarker.task
├── test.py
├── requirements.txt
├── .gitignore
└── README.md
```
