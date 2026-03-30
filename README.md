# Supervisor-AI-Multimodal-Proctoring

Supervisor-AI: Real-Time Multimodal Proctoring Framework
📌 Overview

With the rapid adoption of online examinations, maintaining academic integrity has become increasingly challenging. Traditional proctoring solutions are often intrusive, limited in scope, and prone to errors.

Supervisor-AI is a real-time, intelligent multimodal proctoring system that leverages computer vision, audio analysis, and system monitoring to detect suspicious behavior during online exams.

🚀 Key Features
🎥 Computer Vision Monitoring
Face Detection & Tracking
Eye Gaze Tracking
Head Pose Estimation
Blink Detection
Background Motion Detection
Multi-person Detection
📱 Object Detection
YOLO-based detection for:
Mobile phones
Unauthorized objects
🎙️ Audio Intelligence
Voice Activity Detection (VAD)
Multi-speaker detection
External voice anomaly detection
💻 System Activity Tracking
Tab switching detection
Window minimization monitoring
Copy-paste activity tracking
🧠 AI-Based Decision System
Multimodal signal fusion
Real-time Suspicion Score Generation
Machine Learning aggregation model
🔍 Explainable AI (XAI)
Transparent alerts
Highlights contributing factors:
Gaze deviation
Phone detection
Multiple voices

[ Webcam Feed ] ─┐
                 ├──> [ Computer Vision Module ]
[ Microphone ] ──┤
                 ├──> [ Audio Analysis Module ]
[ System Logs ] ─┘

        ↓

[ Feature Fusion Layer ]
        ↓
[ ML Suspicion Model ]
        ↓
[ Explainable AI Output ]
        ↓
[ Real-Time Alerts Dashboard ]

| Domain            | Technologies Used            |
| ----------------- | ---------------------------- |
| Computer Vision   | OpenCV, MediaPipe            |
| Object Detection  | YOLO (Deep Learning)         |
| Audio Processing  | Python Audio Libraries, VAD  |
| Machine Learning  | Scikit-learn / Custom Models |
| Backend Logic     | Python                       |
| System Monitoring | OS-level Event Tracking      |

How It Works
Captures video, audio, and system activity
Extracts behavioral features:
Eye movement
Head orientation
Background changes
Detects anomalies using AI models
Combines all signals into a single suspicion score
Provides real-time alerts with explanations
🎯 Applications
Online examinations (colleges, universities)
Remote hiring assessments
Certification platforms
Corporate compliance testing
⚡ Highlights
✅ Real-time detection
✅ Multimodal intelligence (Video + Audio + System)
✅ Lightweight & scalable
✅ Privacy-aware design
✅ Explainable AI integration

Supervisor-AI/
│── src/
│   ├── vision/
│   ├── audio/
│   ├── system_monitor/
│   ├── fusion_model/
│
│── models/
│── data/
│── notebooks/
│── results/
│
│── app.py
│── requirements.txt
│── README.md

Future Improvements
🔹 Web dashboard for live monitoring
🔹 Cloud-based deployment
🔹 Advanced deep learning models
🔹 Better privacy-preserving AI
🔹 Integration with LMS platforms

👨‍💻 Team
Narashimamurthy K
Shreyas J
Skanda BS
