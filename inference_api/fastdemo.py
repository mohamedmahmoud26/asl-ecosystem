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

CONFIDENCE_THRESHOLD = 0.6
TARGET_FRAMES = 30
STABLE_FRAMES = 3

# إعدادات الحركة
MOTION_THRESHOLD = 0.02      # الحد الأدنى للحركة عشان نعتبر في إشارة
NO_MOTION_FRAMES = 10        # عدد فريمات السكون قبل ما نقطع الإشارة
MIN_SEQUENCE_FRAMES = 15     # أقل عدد فريمات عشان نبدأ نتوقع

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="ASL Sentence API FINAL")

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

        data = response.json()
        result = data["choices"][0]["message"]["content"].strip()

        # منع الهبد - لو الكلمات مش نفسها ارجع الكلمات زي ما هي
        result_words = result.lower().split()
        if sorted(result_words) != sorted([w.lower() for w in words]):
            return " ".join(words)

        return result


# ================= CORE =================
def process_video(file_path):
    load_model()

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise Exception("Video could not be opened")

    words = []
    history = []
    last_word = None

    sequence = []
    prev_hand_landmarks = None
    no_motion_count = 0
    is_signing = False

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
            # لو الشكل اتغير (ظهرت إيد أو اختفت) اعتبرها حركة
            return 1.0
        return float(np.mean(np.abs(curr - prev)))

    def predict_from_sequence(seq):
        """خد sequence وارجع الكلمة والـ confidence"""
        if len(seq) < MIN_SEQUENCE_FRAMES:
            return None, 0

        # لو أقل من TARGET_FRAMES، اعمل padding بأول فريم
        if len(seq) < TARGET_FRAMES:
            pad = [seq[0]] * (TARGET_FRAMES - len(seq))
            seq_padded = pad + list(seq)
        else:
            seq_padded = seq[-TARGET_FRAMES:]

        input_data = np.expand_dims(np.array(seq_padded, dtype=np.float32), axis=0)

        interpreter.resize_tensor_input(input_index, input_data.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_index)
        probs = np.squeeze(output)
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        return pred, conf

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

            curr_hand = get_hand_center(results)
            motion = compute_motion(prev_hand_landmarks, curr_hand)
            prev_hand_landmarks = curr_hand

            if motion > MOTION_THRESHOLD:
                # ✋ في حركة → ابدأ أو كمل تسجيل الإشارة
                is_signing = True
                no_motion_count = 0
                landmarks = extract_landmarks(results)
                sequence.append(landmarks)

            elif is_signing:
                # 🤚 السكون بدأ بعد إشارة → كمل شوية وبعدين توقع
                no_motion_count += 1
                landmarks = extract_landmarks(results)
                sequence.append(landmarks)

                if no_motion_count >= NO_MOTION_FRAMES:
                    # ✅ الإشارة خلصت → ابدأ التوقع
                    pred, conf = predict_from_sequence(sequence)

                    if pred is not None and conf > CONFIDENCE_THRESHOLD:
                        word = idx_to_sign.get(pred, str(pred))

                        history.append(pred)
                        if len(history) > STABLE_FRAMES:
                            history = history[-STABLE_FRAMES:]

                        # ضيف الكلمة بس لو مش تكرار
                        if word != last_word:
                            words.append(word)
                            last_word = word

                    # 🔄 Reset عشان الإشارة الجاية
                    sequence = []
                    history = []
                    is_signing = False
                    no_motion_count = 0

        # لو الفيديو خلص وفيه sequence لسه معلقة (إشارة من غير سكون في الآخر)
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