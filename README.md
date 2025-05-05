# Person Tracking with YOLOv4-Tiny 🧑‍🤝‍🧑

A real-time person tracking system using YOLOv4-Tiny for object detection. This project helps to track individuals through video streams, providing information on their entry, exit, and time spent in the frame. The application is designed to be simple and efficient for real-time video analysis.

This idea can be extended to cafes, libraries, and other public spaces to track the number of people present at any given time. It can be used for crowd management, ensuring compliance with capacity limits, or simply for analyzing the flow of visitors.

**Technologies Used**: Python, OpenCV, YOLO, NumPy

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Details](#technical-details)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Acknowledgments](#acknowledgments)

---

## 🌱 Overview

This project uses YOLOv4-Tiny to detect and track people in real-time from a video feed. It assigns unique IDs to each detected individual and tracks their movements, including entry and exit from the frame, as well as their total time spent within the view. The system also includes notifications (audio alerts) when individuals enter or exit.

The concept can be applied in cafes, libraries, and other public spaces, allowing businesses and institutions to track the number of visitors in real-time. This can help manage space occupancy, analyze visitor flow, or even optimize staff allocation based on the number of people present.

![Person Tracking Demo](assets/person_tracking_demo.png)

- **Real-time detection**: Tracks people as they appear and disappear.
- **ID tracking**: Each person is assigned a unique ID and tracked across frames.
- **Sound notifications**: Alerts are triggered when a person enters or exits the frame.
- **Applications in cafes and libraries**: Manage visitor flow, ensure capacity limits, or analyze traffic patterns.

---

## ✨ Features

| Feature                   | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| 🚀 **Real-time Analysis**   | Get instant person tracking results                     |
| 🧠 **ID Tracking**          | Unique IDs are assigned to each detected person         |
| 🎯 **Time Tracking**        | Tracks the duration each person stays in the frame      |
| 🔊 **Sound Notifications**  | Audio alerts are triggered on entry/exit events         |
| 💻 **Real-time Display**    | Visual tracking with bounding boxes and timestamps      |
| 📊 **Statistics**           | Displays total persons detected and current visitors    |

---

## 🛠️ Technical Details

### Requirements

- **Python 3.x**
- **OpenCV** (for video processing)
- **NumPy** (for numerical operations)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/person-tracking-yolov4-tiny.git


1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/person-tracking-yolov4-tiny.git



2. **Navigate to the project directory:**

   ```bash
   cd person-tracking-yolov4-tiny
   ```

3. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Download YOLOv4-Tiny files (weights and config):**

   The `yolov4-tiny.weights` and `yolov4-tiny.cfg` files can be downloaded from:

   * [YOLOv4-Tiny Weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)
   * [YOLOv4-Tiny CFG](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)

   Place these files in the project root directory.

---

## 🚀 Usage

1. **Run the script:**

   ```bash
   python main.py
   ```

2. **The program will open your webcam and start detecting and tracking people.** It will display:

   * Bounding boxes around detected individuals.
   * Their unique ID and time spent in the frame.

3. **Exit the application** by pressing the `q` key.

---

## 🧠 Model Architecture

The model uses a pre-trained **YOLOv4-Tiny** network for detecting people. The architecture consists of the following layers:

* **Input Layer**: 416x416x3 (RGB)
* **Multiple Convolutional Layers**: For feature extraction.
* **Max-Pooling Layers**: For down-sampling.
* **Dense Layers**: For classification and regression.
* **Softmax Output Layer**: For multi-class classification, detecting 'person' class.

---

## 📚 Dataset

This project uses a **pre-trained YOLOv4-Tiny** model, which was trained on the **COCO dataset**, specifically focusing on the **person** class for detecting individuals in the frame.

### YOLO Configuration:

* **Input size**: 416x416
* **Classes**: Primarily focuses on human detection (Class: person)

The model is efficient for real-time processing, capable of handling multiple objects with reasonable accuracy and speed.

---

## 🎯 Project Pipeline

1. **Capture Video**: Stream video input from the webcam.
2. **Person Detection**: Use YOLOv4-Tiny for detecting people in each frame.
3. **ID Assignment**: Assign unique IDs to each detected person using a simple tracker.
4. **Tracking & Notification**: Track the movement of each person across frames and provide audio notifications on entry and exit.
5. **Statistics**: Display the total number of people detected and currently present in the frame.

---

## 🤝 Acknowledgments

* **YOLOv4-Tiny**: Pre-trained model from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).
* **OpenCV**: For real-time video processing.
* **NumPy**: For numerical calculations.
* **The Python community**: For all the open-source libraries and resources.

---


```
