# main.py - AI Focus Monitor (Extended + Voice Assistant + Clean Logging)

import time
import csv
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import os
import warnings

# --- Silence TensorFlow / Mediapipe console spam ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
import logging as py_logging
py_logging.getLogger("absl").setLevel(py_logging.ERROR)

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Sound alerts (beep)
# -------------------------------------------------
try:
    import sounddevice as sd
    HAVE_SOUND = True
except ImportError:
    HAVE_SOUND = False
    logger.warning("sounddevice not available. Audio alerts disabled.")

# -------------------------------------------------
# Voice assistant (TTS)
# -------------------------------------------------
try:
    import pyttsx3
    TTS_ENGINE = pyttsx3.init()
    HAVE_TTS = True
    logger.info("Voice assistant enabled (pyttsx3 loaded)")
except Exception:
    HAVE_TTS = False
    TTS_ENGINE = None
    logger.warning("pyttsx3 not available. Voice assistant disabled.")

# -------------------------------------------------
# Config
# -------------------------------------------------
try:
    import focus_config_calibrated as cfg
    logger.info("Using calibrated configuration")
except ImportError:
    import focus_config as cfg
    logger.info("Using default configuration")

from focus_utils import *
from report_plot import make_report


# -------------------------------------------------
# UI + Utility
# -------------------------------------------------
def draw_badge(img, text, color, x=20, y=40):
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), _ = cv2.getTextSize(text, font, 0.6, 2)
        cv2.rectangle(img, (x-10, y-h-10), (x+w+10, y+10), (255,255,255), -1)
        cv2.rectangle(img, (x-10, y-h-10), (x+w+10, y+10), (0,0,0), 2)
        cv2.putText(img, text, (x, y), font, 0.6, color, 2, cv2.LINE_AA)
    except:
        pass


def play_alert_sound():
    if not HAVE_SOUND: return
    try:
        t = np.linspace(0, 0.3, int(44100 * 0.3), False)
        wave = (np.sin(2*np.pi*880*t) * np.iinfo(np.int16).max * 0.4).astype(np.int16)
        sd.play(wave, 44100)
    except:
        pass


def speak(text):
    if not HAVE_TTS: return
    try:
        TTS_ENGINE.stop()
        TTS_ENGINE.say(text)
        TTS_ENGINE.runAndWait()
    except:
        pass


def validate_config():
    if cfg.EAR_SLEEPY_THRESHOLD <= cfg.EAR_BLINK_THRESHOLD:
        raise ValueError("EAR threshold error")


def initialize_camera(index=0):
    cap = cv2.VideoCapture(index)
    cap.set(3, 640)
    cap.set(4, 480)
    if not cap.isOpened():
        raise RuntimeError("Camera failed")
    return cap


# -------------------------------------------------
# Core frame processing
# -------------------------------------------------
def process_frame_metrics(results, frame_shape, ear_buffer):
    h, w = frame_shape[:2]
    metrics = {"ear":0,"mar":0,"yaw":1,"pitch":1}

    if not results.multi_face_landmarks:
        return metrics, False

    lm = results.multi_face_landmarks[0].landmark

    raw_ear = (eye_aspect_ratio(lm, LEFT_EYE, w, h) + eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2
    ear_buffer.add(raw_ear)
    metrics["ear"] = ear_buffer.get_average()

    metrics["mar"] = mouth_aspect_ratio(lm, w, h)
    metrics["yaw"] = compute_yaw_proxy(lm, w, h)
    metrics["pitch"] = compute_pitch_proxy(lm, w, h)

    return metrics, True


def determine_focus_state(metrics, counters, face_detected):
    if not face_detected:
        counters["noface"] += 1
        return "No Face"

    counters["noface"] = 0

    if metrics["mar"] > cfg.MAR_YAWN_THRESHOLD: return "Yawning"
    if 0 < metrics["ear"] < cfg.EAR_BLINK_THRESHOLD: return "Blinking"

    if metrics["ear"] < cfg.EAR_SLEEPY_THRESHOLD:
        counters["sleepy"] += 1
        if counters["sleepy"] > cfg.SLEEPY_FRAME_COUNT: return "Sleepy"
    else:
        counters["sleepy"] = 0

    if (metrics["yaw"] > cfg.YAW_DISTRACT_THRESHOLD_RATIO or
        metrics["pitch"] > cfg.PITCH_DISTRACT_THRESHOLD_RATIO_HIGH):
        counters["distracted"] += 1
        if counters["distracted"] > cfg.DISTRACT_FRAME_COUNT: return "Distracted"
    else:
        counters["distracted"] = 0

    return "Focused"


# -------------------------------------------------
# Main app
# -------------------------------------------------
def main():

    validate_config()
    cap = initialize_camera(cfg.CAM_INDEX)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"session_log_{session_id}.csv"

    ear_buffer = EARBuffer(window_size=cfg.EAR_SMOOTH_WINDOW)
    counters = {"sleepy":0,"distracted":0,"noface":0}

    total_frames, focused_frames = 0, 0
    blink_count, yawn_count = 0, 0

    last_voice_time = 0
    prev_state = "No Face"
    start_time = time.time()

    colors = {
        "Focused": (0,255,0), "Distracted": (0,165,255),
        "Sleepy": (0,0,255), "Blinking": (255,125,0),
        "Yawning": (255,0,255), "No Face": (128,128,128)
    }

    with open(csv_filename, "w", newline="") as f, face_mesh_detector() as fm:
        writer = csv.writer(f)
        writer.writerow(["t_sec","status","ear","mar","yaw_ratio","pitch_ratio"])

        while True:
            ret, frame = cap.read()
            if not ret: continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = fm.process(rgb)
            metrics, face_detected = process_frame_metrics(results, frame.shape, ear_buffer)

            state = determine_focus_state(metrics, counters, face_detected)

            if state == "Blinking" and prev_state != "Blinking": blink_count += 1
            if state == "Yawning" and prev_state != "Yawning": yawn_count += 1
            prev_state = state

            total_frames += 1
            if state == "Focused": focused_frames += 1

            # Voice Assistant
            if state in ["Sleepy","Distracted","No Face"] and time.time() - last_voice_time > cfg.VOICE_COOLDOWN_SEC:
                speak("Stay focused." if state=="Distracted" else 
                      "You look sleepy." if state=="Sleepy" else 
                      "Face not detected.")
                last_voice_time = time.time()

            score = (focused_frames/total_frames)*100
            draw_badge(frame, f"{state} ({score:.1f}%)", colors[state])

            writer.writerow([round(time.time()-start_time,2),state,metrics["ear"],metrics["mar"],metrics["yaw"],metrics["pitch"]])

            cv2.imshow("AI Focus Monitor", frame)
            key = cv2.waitKey(1)

            if key == ord("q") or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    # ----------- Report Generation ----------- #
    try:
        report_png = f"focus_report_{session_id}.png"
        make_report(csv_filename, out_png=report_png)
        logger.info(f"Report saved to {report_png}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")


if __name__ == "__main__":
    main()