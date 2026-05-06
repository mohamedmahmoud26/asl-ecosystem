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

# ==============================================================================
# CONFIG
# ==============================================================================

MODEL_PATH = "../artifacts/tflite/combined_model.tflite"

LABEL_MAP_PATH = "../artifacts/tflite/sign_to_prediction_index_map.json"
CONFIDENCE_THRESHOLD = 0.30
MIN_FRAMES_FOR_SIGN = 5

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==============================================================================
# APP
# ==============================================================================

app = FastAPI(title="ASL API")

# ==============================================================================
# LOAD MODEL
# ==============================================================================

print("[INFO] Loading label map...")

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:

    label_data = json.load(f)

    label_to_sign = {
        int(v): k for k, v in label_data.items()
    }

print("[INFO] Loading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

_cached_shape = None

# ==============================================================================
# MEDIAPIPE
# ==============================================================================

mp_holistic = mp.solutions.holistic


def extract_landmarks(results):

    if results.face_landmarks:
        face = np.array(
            [[res.x, res.y, res.z]
             for res in results.face_landmarks.landmark],
            dtype=np.float32
        )
    else:
        face = np.full((468, 3), np.nan, dtype=np.float32)

    if results.left_hand_landmarks:
        lh = np.array(
            [[res.x, res.y, res.z]
             for res in results.left_hand_landmarks.landmark],
            dtype=np.float32
        )
    else:
        lh = np.full((21, 3), np.nan, dtype=np.float32)

    if results.pose_landmarks:
        pose = np.array(
            [[res.x, res.y, res.z]
             for res in results.pose_landmarks.landmark],
            dtype=np.float32
        )
    else:
        pose = np.full((33, 3), np.nan, dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array(
            [[res.x, res.y, res.z]
             for res in results.right_hand_landmarks.landmark],
            dtype=np.float32
        )
    else:
        rh = np.full((21, 3), np.nan, dtype=np.float32)

    return np.concatenate([face, lh, pose, rh])

# ==============================================================================
# LLM MEMORY
# ==============================================================================

conversation_memory = []


# ==============================================================================
# LLM
# ==============================================================================

async def build_sentence_with_groq(words):

    global conversation_memory

    if not words:
        return ""

    if len(words) < 2:
        return " ".join(words)

    if not GROQ_API_KEY:
        return " ".join(words)

    # آخر سياق فقط
    previous_context = "\n".join(
        conversation_memory[-3:]
    )

    prompt = f"""
You are an assistive AI for sign language translation.

Your job is ONLY to reorder words into a natural sentence.

IMPORTANT RULES:
- Use ONLY the given words
- Do NOT add new words
- Do NOT remove words
- Do NOT change words
- Preserve the exact meaning
- Keep the sentence natural and understandable
- Use previous conversation context only to improve ordering

Previous context:
{previous_context}

Current words:
{words}

Return ONLY the reordered sentence.
"""

    async with httpx.AsyncClient(timeout=30.0) as client:

        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",

                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You reorder sign-language words "
                            "into natural sentences without "
                            "adding or removing words."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                # قليل جدًا عشان يقلل الهبد
                "temperature": 0.05,

                "top_p": 0.2,
            },
        )

        data = response.json()

        try:

            result = data["choices"][0]["message"]["content"].strip()

            # حماية:
            # لو أضاف كلمات نرجع الأصل

            original_words = sorted(
                [w.lower() for w in words]
            )

            result_words = sorted(
                result.lower().split()
            )

            if original_words != result_words:

                return " ".join(words)

            # حفظ السياق
            conversation_memory.append(result)

            # حافظ على آخر 5 جمل فقط
            if len(conversation_memory) > 5:
                conversation_memory = conversation_memory[-5:]

            return result

        except:

            return " ".join(words)
# ==============================================================================
# PROCESS VIDEO
# ==============================================================================

def process_video(file_path):

    global _cached_shape

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise Exception("Cannot open video")

    sequence = []

    sentence = []

    is_signing = False

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = holistic.process(rgb)

            hands_detected = bool(
                results.left_hand_landmarks or
                results.right_hand_landmarks
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
                            np.array(sequence, dtype=np.float32),
                            axis=0
                        )

                        print("INPUT SHAPE:", input_data.shape)

                        # dynamic resize
                        if input_data.shape != _cached_shape:

                            _cached_shape = input_data.shape

                            interpreter.resize_tensor_input(
                                input_details["index"],
                                input_data.shape
                            )

                            interpreter.allocate_tensors()

                        interpreter.set_tensor(
                            input_details["index"],
                            input_data
                        )

                        interpreter.invoke()

                        output_data = interpreter.get_tensor(
                            output_details["index"]
                        )

                        probabilities = np.squeeze(output_data)

                        pred_index = int(np.argmax(probabilities))

                        confidence = float(
                            probabilities[pred_index]
                        )

                        print("CONF:", confidence)

                        if confidence > CONFIDENCE_THRESHOLD:

                            sign_name = label_to_sign[pred_index]

                            sentence.append(sign_name)

                            print("WORD:", sign_name)

    cap.release()

    return sentence


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/")
def home():

    return {
        "status": "running"
    }


@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...)
):

    temp_path = None

    try:

        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp.write(await file.read())

        temp.close()

        temp_path = temp.name

        words = process_video(temp_path)

        sentence = await build_sentence_with_groq(words)

        return {
            "success": True,
            "words": words,
            "sentence": sentence
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)