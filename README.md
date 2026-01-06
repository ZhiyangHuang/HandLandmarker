# MediaPipe Hand Gesture Recognition with Google AI Edge

This project demonstrates real-time **hand gesture recognition** and **hand landmark detection** using **Google AI Edge 2025 library** and **MediaPipe**. It provides a Python implementation for hands-on practice and can be extended to other recognition tasks such as **face landmark detection** and **pose landmark detection**.

---

## Features

- Detect multiple hands in real-time from a camera feed
- Identify gestures using Gesture Recognizer
- Display hand landmarks and skeleton
- Support Left / Right hand identification
- Depth-based visualization using Z-coordinate
- Easily extendable to other MediaPipe tasks (Face / Pose landmarks)

---

## Requirements

- Python 3.10+  
- OpenCV (`cv2`)  
- MediaPipe AI Edge 2025  

---

## 1️⃣ Install Python Dependencies

1. **Install OpenCV**:

```bash
pip install opencv-python
```

2. **Install MediaPipe AI Edge**:
For the 2025 version:

```bash
pip install mediapipe
```

## 2️⃣ Download Model Files (`.task`)

MediaPipe tasks require pre-trained **`.task` files**. You need at least:

| Task Name            | Description                 | Official Download Link |
|----------------------|-----------------------------|----------------------|
| Hand Landmarker      | Detect 21 hand keypoints and handedness | [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) |
| Gesture Recognizer   | Recognize hand gestures (Thumbs Up, Open Palm, etc.) | [gesture_recognizer.task](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task) |
| Face Landmarker      | Detect face landmarks | Visit [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python) |
| Pose Landmarker      | Detect pose landmarks | Visit [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) |

**Instructions:**

1. Click the links to download the `.task` files.  
2. Place the files in your project folder, e.g., `models/`:

or **Use the model task file that I have already prepared.**:
```gesture_recognizer.task```
```hand_landmarker.task```

📌 **Tip:** This project demonstrates practical use of **Google 2025 AI Edge library** with **MediaPipe** for **hand gesture recognition** and **hand landmark detection**. After the 2025 restructure, each tool is an independent module, replacing the previous `mediapipe.solutions` library. The Python implementation helps users quickly learn MediaPipe and extend to other tasks such as **face** and **pose landmark detection**.  

The `main.py` example in this project specifically demonstrates the use of **GestureRecognizer** and **HandLandmarker**, providing a practical reference for real-time camera inference and task integration.

## 📘 Learn More From Official MediaPipe Guide

To deepen your understanding of MediaPipe and explore other task models such as **Face Landmark Detection**, **Pose Landmark Detection**, **Object Detection**, and more, we recommend reading the official Google AI Edge MediaPipe guide:

➡️ https://ai.google.dev/edge/mediapipe/solutions/guide

This guide provides authoritative documentation on the design, usage patterns, configuration options, and best practices for all MediaPipe vision tasks. It explains how to set up tasks in **IMAGE**, **VIDEO**, and **LIVE_STREAM** running modes, how to handle asynchronous callbacks, and how to interpret model outputs like keypoints, bounding boxes, and classification results.

By studying the official MediaPipe guide in combination with the provided `main.py` example in this project, you can:

- Quickly understand the general **API structure** shared by different tasks.
- Learn how to **load and run models** for various scenarios.
- Adapt the documented usage patterns to implement your own recognition pipelines in Python.
- Expand your project to support other recognition tools such as **Face**, **Pose**, **Object Tracking**, and more.

Learn from the official guide and apply it immediately using this project’s sample code for faster mastery of MediaPipe tasks.
