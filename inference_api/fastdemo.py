from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf   #
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

CONFIDENCE_THRESHOLD = 0.30
MIN_FRAMES_FOR_SIGN = 5
GROQ_API_KEY = os.getenv("api_groq")

app = FastAPI(title="ASL Real-Time Style API")

# ================= GLOBAL MODEL =================
interpreter = None
input_index = None
output_index = None
idx_to_sign = None


def load_model():
    global interpreter, input_index, output_index, idx_to_sign

    if interpreter is None:
        print("⏳ Loading model...")

        # ✅ الحل هنا
        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH
        )

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_index = input_details[0]['index']
        output_index = output_details[0]['index']

        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)

        idx_to_sign = {v: k for k, v in label_map.items()}

        print("✅ Model ready")


# ================= MEDIAPIPE =================
mp_holistic = mp.solutions.holistic


def extract_landmarks(results):
    def to_arr(lms, n):
        if lms:
            return np.array([[l.x, l.y, l.z] for l in lms.landmark], dtype=np.float32)
        return np.full((n, 3), np.nan, dtype=np.float32)

    face = to_arr(results.face_landmarks, 468)
    lh = to_arr(results.left_hand_landmarks, 21)
    pose = to_arr(results.pose_landmarks, 33)
    rh = to_arr(results.right_hand_landmarks, 21)

    return np.concatenate([face, lh, pose, rh])


# ================= GROQ =================
async def build_sentence_with_groq(words: list[str]) -> str:
    if not words:
        return ""

    if not GROQ_API_KEY:
        return " ".join(words)

    prompt = (
        f"The following words were detected from ASL signs: {', '.join(words)}. "
        "Construct a single natural, grammatically correct English sentence using these words. "
        "Return ONLY the sentence, no explanation."
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
                "max_tokens": 100,
                "temperature": 0.3,
            },
            timeout=10.0,
        )

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


# ================= CORE PROCESS =================
def process_video(file_path):
    load_model()

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise Exception("Video could not be opened")

    sequence = []
    sentence = []
    is_signing = False
    _cached_shape = None

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

            hands_detected = bool(
                results.left_hand_landmarks or results.right_hand_landmarks
            )

            landmarks = extract_landmarks(results)

            if hands_detected:
                if not is_signing:
                    is_signing = True
                    sequence = []

                sequence.append(landmarks)

            else:
                if is_signing:
                    is_signing = False

                    if len(sequence) >= MIN_FRAMES_FOR_SIGN:
                        input_data = np.expand_dims(
                            np.array(sequence, dtype=np.float32), axis=0
                        )

                        if input_data.shape != _cached_shape:
                            _cached_shape = input_data.shape
                            interpreter.resize_tensor_input(
                                input_index, input_data.shape
                            )
                            interpreter.allocate_tensors()

                        interpreter.set_tensor(input_index, input_data)
                        interpreter.invoke()

                        output = interpreter.get_tensor(output_index)

                        probs = np.squeeze(output)
                        pred = int(np.argmax(probs))
                        conf = float(probs[pred])

                        if conf > CONFIDENCE_THRESHOLD:
                            word = idx_to_sign.get(pred, str(pred))
                            sentence.append(word)

                            if len(sentence) > 6:
                                sentence.pop(0)

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