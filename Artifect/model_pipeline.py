import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from google import genai
from tensorflow.keras.models import load_model

# -------------------------
# CONFIG
# -------------------------

TARGET_FRAMES = 50
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "1D_CNN_multi_output_LSTM.keras")

# -------------------------
# GLOBAL MODELS (lazy load)
# -------------------------

movenet_model = None
model = None
gemini_model = None


def load_all_models():
    global movenet_model, model, gemini_model

    # Load MoveNet
    if movenet_model is None:
        print("Loading MoveNet...")
        movenet = hub.load(
            "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        )
        movenet_model = movenet.signatures["serving_default"]

    # Load your trained model
    if model is None:
        print("Loading LSTM model...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = load_model(MODEL_PATH)

    # Load Gemini
    if gemini_model is None:
        print("Initializing Gemini...")
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            print("GOOGLE_API_KEY not set. Gemini disabled.")
            gemini_model = None
        else:
            gemini_model = genai.Client(api_key=api_key)


# -------------------------
# STEP 1: Normalize video
# -------------------------

def video_to_50_frames(video_path):
    if not os.path.exists(video_path):
        raise ValueError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    frames = np.array(frames)
    num_frames = len(frames)

    if num_frames > TARGET_FRAMES:
        idx = np.linspace(0, num_frames - 1, TARGET_FRAMES).astype(int)
        frames = frames[idx]

    elif num_frames < TARGET_FRAMES:
        pad_n = TARGET_FRAMES - num_frames
        last_frame = frames[-1]
        pad_frames = np.repeat(last_frame[np.newaxis, ...], pad_n, axis=0)
        frames = np.concatenate([frames, pad_frames], axis=0)

    return frames


# -------------------------
# STEP 2: MoveNet
# -------------------------

def run_movenet(frame_rgb):
    img = tf.image.resize_with_pad(
        tf.expand_dims(frame_rgb, axis=0),
        256, 256
    )
    img = tf.cast(img, tf.int32)

    outputs = movenet_model(img)
    keypoints = outputs["output_0"].numpy()[0, 0, :, :2]

    return keypoints  # (17, 2)


# -------------------------
# STEP 3: Extract keypoints
# -------------------------

def extract_keypoints_from_video(video_path):
    frames = video_to_50_frames(video_path)

    keypoints_sequence = []

    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = run_movenet(frame_rgb)
        keypoints_sequence.append(keypoints.flatten())

    return np.array(keypoints_sequence)  # (50, 34)


# -------------------------
# STEP 4: Labels
# -------------------------

STYLE_LABELS = ["Back", "Front", "Goblet"]

POSTURE_LABELS = [
    "Correct",
    "Insufficient_depth",
    "Rounded_back",
    "Weight_on_toes"
]

STYLE_FEEDBACK_TEMPLATES = {
    "Back": "You are performing a back squat.",
    "Front": "You are performing a front squat.",
    "Goblet": "You are performing a goblet squat."
}

POSTURE_FEEDBACK_TEMPLATES = {
    "Correct": "Your squat technique is correct.",
    "Insufficient_depth": "You are not reaching sufficient squat depth.",
    "Rounded_back": "Your lower back is rounding. Keep a neutral spine.",
    "Weight_on_toes": "Your weight is shifting toward your toes. Keep pressure on your heels."
}


def build_feedback_template(style, posture):
    return (
        STYLE_FEEDBACK_TEMPLATES.get(style, "")
        + " "
        + POSTURE_FEEDBACK_TEMPLATES.get(posture, "")
    )


# -------------------------
# STEP 5: Gemini
# -------------------------

def rephrase_feedback_with_gemini(template_text):
    if gemini_model is None:
        return template_text

    try:
        prompt = f"""
Rewrite this into ONE short sentence (max 12 words).
No explanations.

{template_text}
"""

        response = gemini_model.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )

        text = response.text.strip() if response.text else template_text

        return text  

    except Exception as e:
        print("Gemini error:", e)
        return template_text

# -------------------------
# MAIN FUNCTION
# -------------------------

def process_video(video_path):
    # Load models safely
    load_all_models()

    # Extract keypoints
    X = extract_keypoints_from_video(video_path)

    # Shape check
    if X.shape != (50, 34):
        raise ValueError(f"Unexpected input shape: {X.shape}")

    X = X.reshape(1, 50, 34)

    # Predict
    preds = model.predict(X)

    if not isinstance(preds, (list, tuple)) or len(preds) != 2:
        raise ValueError("Model output format unexpected")

    style_pred, posture_pred = preds

    style_idx = int(np.argmax(style_pred))
    posture_idx = int(np.argmax(posture_pred))

    predicted_style = STYLE_LABELS[style_idx]
    predicted_posture = POSTURE_LABELS[posture_idx]

    # Feedback
    template_text = build_feedback_template(
        predicted_style,
        predicted_posture
    )

    final_feedback = rephrase_feedback_with_gemini(template_text)

    return {
        "style": predicted_style,
        "posture": predicted_posture,
        "feedback": final_feedback
    }