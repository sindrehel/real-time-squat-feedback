# Real-Time Squat Feedback

A computer vision and machine learning system for automated squat
analysis and feedback generation.

This project was developed as my Bachelor's thesis in Applied Data
Science at Noroff University College.

## Overview

The system analyzes squat videos using human pose estimation and a
trained deep learning model.

The application extracts human body keypoints from video frames using
MoveNet Thunder. The resulting keypoint sequence is then passed to a
multi-output deep learning model that predicts:

- Squat style
- Squat posture

The predictions are converted into predefined feedback and optionally
rephrased using Google's Gemini API.

## System Architecture

```text
Video
  │
  ▼
FastAPI
  │
  ▼
OpenCV
  │
  ▼
50 video frames
  │
  ▼
MoveNet Thunder
  │
  ▼
17 body keypoints × 2 coordinates
  │
  ▼
50 × 34 keypoint sequence
  │
  ▼
CNN / CNN-LSTM model
  │
  ├───────────────┐
  ▼               ▼
Squat Style     Posture
  │               │
  └───────┬───────┘
          ▼
   Feedback Template
          │
          ▼
     Google Gemini
          │
          ▼
   Natural Language
      Feedback
```

## Features
- Video upload through a FastAPI backend
- Human pose estimation using MoveNet Thunder
- Fixed-length 50-frame input representation
- Pose keypoint extraction
- Multi-output squat classification
- Squat style classification
- Posture classification
- Automated feedback generation
- Optional Google Gemini integration

## Classification
Squat -
The model predicts between three squat styles:
- Back squat
- Front squat
- Goblet squat
  
Posture -
The model predicts four posture categories:
- Correct
- Insufficient depth
- Rounded back
- Weight on toes

## Technologies
Programming:
- Python
  
Computer Vision:
- OpenCV
- MoveNet Thunder
  
Machine Learning:
- TensorFlow
- Keras
- TensorFlow Hub
- CNN
- LSTM
- NumPy
  
Backend:
- FastAPI
- Uvicorn
  
Feedback Generation:
- Google Gemini API

## Project Structure

## How It Works

The system processes an uploaded squat video through several stages, from video preprocessing and pose estimation to machine learning classification and natural-language feedback.

### 1. Video Upload

The user uploads a squat video through the frontend.

The frontend sends the video to the FastAPI `/upload/` endpoint for processing.

### 2. Video Processing

The uploaded video is processed using OpenCV.

Each video is converted into a fixed sequence of **50 frames**:

- Videos with more than 50 frames are sampled down to 50 frames.
- Videos with fewer than 50 frames are padded using the last available frame.

### 3. Pose Estimation

**MoveNet Thunder** is used to extract human pose keypoints from each frame.

Each frame produces **17 body keypoints**, with an `x` and `y` coordinate for each keypoint.

This results in a feature representation with the shape:

```text
(50, 34)
```
where:

50 = number of frames\
34 = 17 keypoints × 2 coordinates (x, y)


### 4. Machine Learning Prediction
The extracted keypoint sequence is passed to the trained multi-output machine learning model.

The model produces two predictions:

Squat style
Posture

For example:
```text
Squat style: Front
Posture: Insufficient Depth
```
### 5. Feedback Generation
   
The predicted classes are mapped to predefined feedback templates.

For example, a prediction of:
```text
Style: Front
Posture: Insufficient Depth
```
is used to generate an appropriate feedback template.

### 6. Natural Language Feedback

If a Gemini API key is configured, the feedback template is sent to Gemini to generate a short, natural-language feedback message.

If Gemini is unavailable, the system falls back to the predefined feedback template.

Example Output
```text
"style": "Front",
"posture": "Insufficient_depth",
"feedback": "Try to reach a deeper squat position."
```
## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/real-time-squat-feedback.git
cd real-time-squat-feedback
```
### 2. Create a virtual environment
```bash
python -m venv venv
```
Windows:
```bash
venv\Scripts\activate
```
macOS / Linux:
```bash
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure the Gemini API key 
The system can use Gemini to generate natural-language feedback.

Set your API key as an environment variable:

Windows:
```bash
set GOOGLE_API_KEY=your_api_key_here
```
macOS / Linux:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

### 5. Start the FastAPI server
```bash
uvicorn app:app
```
The API will be available at:
```bash
http://127.0.0.1:8000
```
### 6. Open the API documentation
FastAPI automatically provides interactive API documentation at:
```bash
http://127.0.0.1:8000/docs
```
From the documentation, you can test the /upload/ endpoint by uploading a video file.

## Results

The bachelor's thesis evaluated both a 1D CNN and a CNN-LSTM
architecture.

The experiments showed that the CNN model performed better than the
CNN-LSTM model during the training and validation experiments.

However, performance decreased when the models were evaluated on
participants who were not represented in the training data.

This highlighted generalization across individuals as a major
limitation of the project.

## Limitations

The dataset used in the project was relatively small and had limited
participant diversity.

Other limitations included:

- Controlled recording environment
- Limited variation in clothing and background
- Side-view recordings
- No loaded barbell during data collection
- Limited number of participants
- 2D pose representation
- Limited generalization to unseen participants

## Bachelor's Thesis

This project was developed as part of my Bachelor's degree in Applied
Data Science at Noroff University College.

The project investigated the use of pose estimation, deep learning and
large language models for automated squat analysis and feedback.

## Author
Sindre Heldal
