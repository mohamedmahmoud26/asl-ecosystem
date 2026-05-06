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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts/tflite/combined_model.tflite"
)

LABEL_MAP_PATH = os.path.join(
    BASE_DIR,
    "artifacts/tflite/sign_to_prediction_index_map.json"
)

CONFIDENCE_THRESHOLD = 0.35
MIN_SEQUENCE_FRAMES = 10

TARGET_FRAMES = 64

# motion
MOTION_THRESHOLD = 0.005
NO_MOTION_FRAMES = 8

# ================= LANDMARKS =================

ROWS_PER_FRAME = 543

LIP = [
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]

NOSE = [1, 2, 98, 327]

LHAND = list(range(468, 489))
RHAND = list(range(522, 543))

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 246, 161, 160, 159, 158, 157, 173
]

LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382,
    362, 466, 388, 387, 386, 385, 384, 398
]

POINT_LANDMARKS = (
    LIP +
    LHAND +
    RHAND +
    NOSE +
    REYE +
    LEYE
)

NUM_NODES = len(POINT_LANDMARKS)

CHANNELS = 6 * NUM_NODES

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================= APP =================

app = FastAPI(title="ASL Sentence API")

# ================= MODEL =================

interpreter = None
input_index = None
output_index = None
idx_to_sign = None
INPUT_SHAPE = None


def load_model():
    global interpreter
    global input_index
    global output_index
    global idx_to_sign
    global INPUT_SHAPE

    if interpreter is None:

        interpreter = tf.lite.Interpreter(
            model_path=MODEL_PATH
        )

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        INPUT_SHAPE = input_details[0]["shape"]

        print("MODEL INPUT SHAPE:", INPUT_SHAPE)

        input_index = input_details[0]["index"]
        output_index = output_details[0]["index"]

        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)

        idx_to_sign = {
            v: k for k, v in label_map.items()
        }

        # allocate مرة واحدة بس
        interpreter.allocate_tensors()


# ================= MEDIAPIPE =================

mp_holistic = mp.solutions.holistic


def extract_landmarks_raw(results):

    def to_arr(lms, n):

        if lms:
            return np.array(
                [[lm.x, lm.y, lm.z] for lm in lms.landmark],
                dtype=np.float32
            )

        return np.full(
            (n, 3),
            np.nan,
            dtype=np.float32
        )

    face = to_arr(results.face_landmarks, 468)
    lh = to_arr(results.left_hand_landmarks, 21)
    pose = to_arr(results.pose_landmarks, 33)
    rh = to_arr(results.right_hand_landmarks, 21)

    return np.concatenate(
        [face, lh, pose, rh],
        axis=0
    )


# ================= PAD =================

def pad_sequence(x):

    T = x.shape[0]

    if T < TARGET_FRAMES:

        pad = np.zeros(
            (
                TARGET_FRAMES - T,
                x.shape[1]
            ),
            dtype=np.float32
        )

        x = np.concatenate([x, pad], axis=0)

    elif T > TARGET_FRAMES:

        idx = np.linspace(
            0,
            T - 1,
            TARGET_FRAMES
        ).astype(np.int32)

        x = x[idx]

    return x


# ================= PREPROCESS =================

def preprocess_sequence(sequence):

    x = np.array(
        sequence,
        dtype=np.float32
    )

    # reference point
    ref = x[:, 17:18, :]

    mean = np.nanmean(
        ref,
        axis=(0, 1),
        keepdims=True
    )

    if np.isnan(mean).all():

        mean = np.array(
            [[[0.5, 0.5, 0.5]]],
            dtype=np.float32
        )

    x_sel = x[:, POINT_LANDMARKS, :]

    std = np.nanstd(
        x_sel - mean,
        axis=(0, 1),
        keepdims=True
    )

    std = np.where(std == 0, 1.0, std)

    x_sel = (x_sel - mean) / std

    # x,y only
    x_sel = x_sel[..., :2]

    T = x_sel.shape[0]

    dx = np.zeros_like(x_sel)
    dx2 = np.zeros_like(x_sel)

    if T > 1:
        dx[:-1] = x_sel[1:] - x_sel[:-1]

    if T > 2:
        dx2[:-2] = x_sel[2:] - x_sel[:-2]

    x_sel = np.nan_to_num(x_sel, nan=0.0)
    dx = np.nan_to_num(dx, nan=0.0)
    dx2 = np.nan_to_num(dx2, nan=0.0)

    x_out = np.concatenate(
        [
            x_sel.reshape(T, -1),
            dx.reshape(T, -1),
            dx2.reshape(T, -1),
        ],
        axis=-1
    )

    x_out = pad_sequence(x_out)

    return x_out[np.newaxis, ...].astype(np.float32)


# ================= LLM =================

async def build_sentence_with_groq(words):

    if not words:
        return ""

    if len(words) < 2:
        return " ".join(words)

    if not GROQ_API_KEY:
        return " ".join(words)

    prompt = f"""
Words:
{words}

Return ONLY a grammatically correct sentence
using EXACTLY these words.

Do not add words.
Do not explain.
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
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
            },
        )

        data = response.json()

        try:

            result = data["choices"][0]["message"]["content"].strip()

            result_words = result.lower().split()

            if sorted(result_words) != sorted(
                [w.lower() for w in words]
            ):
                return " ".join(words)

            return result

        except:
            return " ".join(words)


# ================= PREDICT =================

def predict_from_sequence(seq):

    if len(seq) < MIN_SEQUENCE_FRAMES:
        return None, 0

    input_data = preprocess_sequence(seq)

    interpreter.set_tensor(
        input_index,
        input_data
    )

    interpreter.invoke()

    output = interpreter.get_tensor(output_index)

    probs = np.squeeze(output)

    pred = int(np.argmax(probs))

    conf = float(probs[pred])

    return pred, conf


# ================= MOTION =================

def get_hand_center(results):

    points = []

    for hand in [
        results.left_hand_landmarks,
        results.right_hand_landmarks
    ]:

        if hand:

            for lm in hand.landmark:
                points.append([lm.x, lm.y])

    return np.array(points) if points else None


def compute_motion(prev, curr):

    if prev is None or curr is None:
        return 0

    if prev.shape != curr.shape:
        return 1.0

    return float(
        np.mean(np.abs(curr - prev))
    )


# ================= PROCESS VIDEO =================

def process_video(file_path):

    load_model()

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise Exception("Cannot open video")

    words = []

    last_word = None

    sequence = []

    prev_hand = None

    no_motion_count = 0

    is_signing = False

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = holistic.process(rgb)

            curr_hand = get_hand_center(results)

            motion = compute_motion(
                prev_hand,
                curr_hand
            )

            prev_hand = curr_hand

            if motion > MOTION_THRESHOLD:

                is_signing = True

                no_motion_count = 0

                landmarks = extract_landmarks_raw(results)

                if curr_hand is not None:
                    sequence.append(landmarks)

            elif is_signing:

                no_motion_count += 1

                landmarks = extract_landmarks_raw(results)

                if curr_hand is not None:
                    sequence.append(landmarks)

                if no_motion_count >= NO_MOTION_FRAMES:

                    pred, conf = predict_from_sequence(sequence)

                    print("CONF:", conf)

                    if (
                        pred is not None
                        and conf > CONFIDENCE_THRESHOLD
                    ):

                        word = idx_to_sign.get(
                            pred,
                            str(pred)
                        )

                        if word != last_word:

                            words.append(word)

                            last_word = word

                            print("WORD:", word)

                    # reset
                    sequence = []

                    is_signing = False

                    no_motion_count = 0

        # آخر إشارة
        if sequence and is_signing:

            pred, conf = predict_from_sequence(sequence)

            if (
                pred is not None
                and conf > CONFIDENCE_THRESHOLD
            ):

                word = idx_to_sign.get(
                    pred,
                    str(pred)
                )

                if word != last_word:
                    words.append(word)

    cap.release()

    return words


# ================= ENDPOINT =================

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