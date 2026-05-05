from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import json
import tempfile
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "artifacts/tflite/combined_model.tflite")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "artifacts/tflite/sign_to_prediction_index_map.json")

CONFIDENCE_THRESHOLD = 0.65
TARGET_FRAMES = 30
COOLDOWN_FRAMES = 15

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="ASL Clean API")

# ================= MODEL =================
interpreter = None
input_index = None
output_index = None
idx_to_sign = None


def load_model():
    global interpreter, input_index, output_index, idx_to_sign

    if interpreter is None:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_index = input_details[0]['index']
        output_index = output_details[0]['index']

        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)

        idx_to_sign = {v: k for k, v in label_map.items()}


# ================= MEDIAPIPE =================
mp_holistic = mp.solutions.holistic


def extract_landmarks(results):
    def to_arr(lms, n):
        if lms:
            return np.array([[l.x, l.y, l.z] for l in lms.landmark], dtype=np.float32)
        return np.zeros((n, 3), dtype=np.float32)

    face = to_arr(results.face_landmarks, 468)
    lh = to_arr(results.left_hand_landmarks, 21)
    pose = to_arr(results.pose_landmarks, 33)
    rh = to_arr(results.right_hand_landmarks, 21)

    return np.concatenate([face, lh, pose, rh])


# ================= GROQ =================
async def build_sentence_with_groq(words):
    if not words:
        return ""

    if len(words) < 2 or not GROQ_API_KEY:
        return " ".join(words)

    prompt = (
        f"Words: {words}\n"
        "ONLY rearrange these words into a correct English sentence.\n"
        "DO NOT add, remove, or replace any word.\n"
        "If not possible, return as is."
    )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=10.0,
        )

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


# ================= CORE =================
def process_video(file_path):
    load_model()

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise Exception("Video could not be opened")

    sequence = []
    sentence = []
    last_word = None
    cooldown = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            landmarks = extract_landmarks(results)
            sequence.append(landmarks)

            # Sliding window
            if len(sequence) > TARGET_FRAMES:
                sequence = sequence[-TARGET_FRAMES:]

            if len(sequence) < TARGET_FRAMES:
                continue

            seq = np.array(sequence, dtype=np.float32)
            input_data = np.expand_dims(seq, axis=0)

            interpreter.resize_tensor_input(input_index, input_data.shape)
            interpreter.allocate_tensors()

            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()

            output = interpreter.get_tensor(output_index)

            probs = np.squeeze(output)
            pred = int(np.argmax(probs))
            conf = float(probs[pred])

            # Prediction logic
            if conf > CONFIDENCE_THRESHOLD and cooldown == 0:
                word = idx_to_sign.get(pred, str(pred))

                if word != last_word:
                    sentence.append(word)
                    last_word = word
                    cooldown = COOLDOWN_FRAMES

            if cooldown > 0:
                cooldown -= 1

    cap.release()
    return sentence


# ================= ENDPOINT =================
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    temp_path = None

    try:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(await file.read())
        temp.close()
        temp_path = temp.name

        words = process_video(temp_path)

        sentence = await build_sentence_with_groq(words)

        return {
            "words": words,
            "sentence": sentence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)