from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tflite_runtime.interpreter as tflite
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="ASL API (TFLite Only)")

# ================= LOAD MODEL =================
interpreter = None
input_index = None
output_index = None
idx_to_sign = None


def load_model():
    global interpreter, input_index, output_index, idx_to_sign

    if interpreter is None:
        print("⏳ Loading model...")

        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_index = input_details[0]['index']
        output_index = output_details[0]['index']

        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)

        idx_to_sign = {v: k for k, v in label_map.items()}

        print("✅ Model ready")


# ================= SIMPLE PROCESS =================
def process_video(file_path):
    load_model()

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise Exception("Video could not be opened")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return []

    input_data = np.expand_dims(np.array(frames, dtype=np.float32), axis=0)

    interpreter.resize_tensor_input(input_index, input_data.shape)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_index)

    probs = np.squeeze(output)
    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    if conf > CONFIDENCE_THRESHOLD:
        return [idx_to_sign.get(pred, str(pred))]

    return []


# ================= GROQ =================
async def build_sentence(words):
    if not words:
        return ""

    if not GROQ_API_KEY:
        return " ".join(words)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": " ".join(words)}],
            },
        )

        data = response.json()
        return data["choices"][0]["message"]["content"]


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
        sentence = await build_sentence(words)

        return {
            "words": words,
            "sentence": sentence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)