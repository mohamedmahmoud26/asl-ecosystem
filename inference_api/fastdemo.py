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

MODEL_PATH      = os.path.join(BASE_DIR, "artifacts/tflite/combined_model.tflite")
LABEL_MAP_PATH  = os.path.join(BASE_DIR, "artifacts/tflite/sign_to_prediction_index_map.json")

CONFIDENCE_THRESHOLD = 0.6
MIN_SEQUENCE_FRAMES  = 15
STABLE_FRAMES        = 3

# إعدادات الحركة
MOTION_THRESHOLD = 0.02
NO_MOTION_FRAMES = 10

# ================= LANDMARKS CONFIG =================
# نفس الـ landmarks بالظبط اللي الموديل اتعلم عليهم من الـ notebook
ROWS_PER_FRAME = 543

LIP = [
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]

NOSE  = [1, 2, 98, 327]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
REYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE  = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_NODES       = len(POINT_LANDMARKS)
CHANNELS        = 6 * NUM_NODES  # X,Y * (position + velocity + acceleration)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="ASL Sentence API")

# ================= MODEL =================
interpreter  = None
input_index  = None
output_index = None
idx_to_sign  = None


def load_model():
    global interpreter, input_index, output_index, idx_to_sign

    if interpreter is None:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_index  = input_details[0]['index']
        output_index = output_details[0]['index']

        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)

        idx_to_sign = {v: k for k, v in label_map.items()}


# ================= MEDIAPIPE =================
mp_holistic = mp.solutions.holistic


def extract_landmarks_raw(results):
    """
    استخرج الـ 543 نقطة كاملة بنفس ترتيب الـ parquet files:
      0   : 468  → face
      468 : 489  → left hand
      489 : 522  → pose
      522 : 543  → right hand

    مهم: بنحط NaN لو الـ landmark مش موجود (مش zero!)
    """
    def to_arr(lms, n):
        if lms:
            return np.array([[l.x, l.y, l.z] for l in lms.landmark], dtype=np.float32)
        return np.full((n, 3), np.nan, dtype=np.float32)

    face = to_arr(results.face_landmarks, 468)
    lh   = to_arr(results.left_hand_landmarks, 21)
    pose = to_arr(results.pose_landmarks, 33)
    rh   = to_arr(results.right_hand_landmarks, 21)

    return np.concatenate([face, lh, pose, rh])  # (543, 3)


# ================= PREPROCESS =================
def preprocess_sequence(sequence):
    """
    نفس الـ Preprocess Layer بالظبط اللي اتعلم عليها الموديل:
      1. Center normalization على نقطة 17 (nose tip)
      2. اختيار POINT_LANDMARKS بس
      3. Std normalization
      4. X و Y بس (drop Z)
      5. Velocity (dx) و Acceleration (dx2)
      6. Concatenate → (1, T, CHANNELS)

    Input:  list of frames, كل frame shape (543, 3)
    Output: numpy array shape (1, T, CHANNELS)
    """
    x = np.array(sequence, dtype=np.float32)  # (T, 543, 3)

    # 1. Center على نقطة 17 (nose tip) — نفس الـ notebook
    ref  = x[:, 17:18, :]
    mean = np.nanmean(ref, axis=(0, 1), keepdims=True)
    if np.isnan(mean).all():
        mean = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)

    # 2. اختار اللاندماركس المطلوبة بس
    x_sel = x[:, POINT_LANDMARKS, :]  # (T, NUM_NODES, 3)

    # 3. Std normalization
    std   = np.nanstd(x_sel - mean, axis=(0, 1), keepdims=True)
    std   = np.where(std == 0, 1.0, std)
    x_sel = (x_sel - mean) / std

    # 4. X و Y بس (drop Z)
    x_sel = x_sel[..., :2]  # (T, NUM_NODES, 2)

    # 5. Velocity و Acceleration
    T   = x_sel.shape[0]
    dx  = np.zeros_like(x_sel)
    dx2 = np.zeros_like(x_sel)

    if T > 1:
        dx[:-1] = x_sel[1:] - x_sel[:-1]
    if T > 2:
        dx2[:-2] = x_sel[2:] - x_sel[:-2]

    # 6. Replace NaN بـ 0
    x_sel = np.nan_to_num(x_sel, nan=0.0)
    dx    = np.nan_to_num(dx,    nan=0.0)
    dx2   = np.nan_to_num(dx2,   nan=0.0)

    # 7. Concatenate → (1, T, CHANNELS)
    x_out = np.concatenate([
        x_sel.reshape(T, -1),
        dx.reshape(T, -1),
        dx2.reshape(T, -1),
    ], axis=-1)

    return x_out[np.newaxis, ...].astype(np.float32)  # (1, T, CHANNELS)


# ================= LLM =================
async def build_sentence_with_groq(words):
    if not words:
        return ""

    if len(words) < 2 or not GROQ_API_KEY:
        return " ".join(words)

    prompt = (
        f"Words: {words}\n"
        "Return ONLY a correct sentence using EXACTLY these words.\n"
        "Do NOT add any new words.\n"
        "Do NOT explain.\n"
        "Only return the sentence."
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
        )

        data   = response.json()
        result = data["choices"][0]["message"]["content"].strip()

        # تأكد إن الكلمات نفسها موجودة
        result_words = result.lower().split()
        if sorted(result_words) != sorted([w.lower() for w in words]):
            return " ".join(words)

        return result


# ================= PREDICTION =================
def predict_from_sequence(seq):
    """
    seq: list of raw frames, كل frame shape (543, 3)
    بيرجع (predicted_index, confidence)
    """
    if len(seq) < MIN_SEQUENCE_FRAMES:
        return None, 0

    input_data = preprocess_sequence(seq)  # (1, T, CHANNELS)

    interpreter.resize_tensor_input(input_index, input_data.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_index)
    probs  = np.squeeze(output)
    pred   = int(np.argmax(probs))
    conf   = float(probs[pred])

    return pred, conf


# ================= MOTION HELPERS =================
def get_hand_center(results):
    """استخرج نقاط الإيدين عشان نحسب الحركة"""
    points = []
    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand:
            for lm in hand.landmark:
                points.append([lm.x, lm.y])
    return np.array(points) if points else None


def compute_motion(prev, curr):
    """احسب مقدار الحركة بين فريمين"""
    if prev is None or curr is None:
        return 0
    if prev.shape != curr.shape:
        return 1.0  # ظهرت إيد أو اختفت = حركة
    return float(np.mean(np.abs(curr - prev)))


# ================= CORE =================
def process_video(file_path):
    load_model()

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise Exception("Video could not be opened")

    words     = []
    history   = []
    last_word = None

    sequence        = []
    prev_hand       = None
    no_motion_count = 0
    is_signing      = False

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            curr_hand = get_hand_center(results)
            motion    = compute_motion(prev_hand, curr_hand)
            prev_hand = curr_hand

            if motion > MOTION_THRESHOLD:
                # ✋ في حركة → سجل الفريم
                is_signing      = True
                no_motion_count = 0
                landmarks       = extract_landmarks_raw(results)  # (543, 3) مع NaN
                sequence.append(landmarks)

            elif is_signing:
                # 🤚 سكون بعد إشارة → كمل شوية وبعدين توقع
                no_motion_count += 1
                landmarks        = extract_landmarks_raw(results)
                sequence.append(landmarks)

                if no_motion_count >= NO_MOTION_FRAMES:
                    # ✅ الإشارة خلصت → توقع
                    pred, conf = predict_from_sequence(sequence)

                    if pred is not None and conf > CONFIDENCE_THRESHOLD:
                        word = idx_to_sign.get(pred, str(pred))

                        history.append(pred)
                        if len(history) > STABLE_FRAMES:
                            history = history[-STABLE_FRAMES:]

                        if word != last_word:
                            words.append(word)
                            last_word = word

                    # 🔄 Reset عشان الإشارة الجاية
                    sequence        = []
                    history         = []
                    is_signing      = False
                    no_motion_count = 0

        # لو الفيديو خلص وفيه إشارة معلقة في الآخر
        if sequence and is_signing:
            pred, conf = predict_from_sequence(sequence)
            if pred is not None and conf > CONFIDENCE_THRESHOLD:
                word = idx_to_sign.get(pred, str(pred))
                if word != last_word:
                    words.append(word)

    cap.release()
    return words


# ================= ENDPOINT =================
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    temp_path = None

    try:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(await file.read())
        temp.close()
        temp_path = temp.name

        words    = process_video(temp_path)
        sentence = await build_sentence_with_groq(words)

        return {
            "words":    words,
            "sentence": sentence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)