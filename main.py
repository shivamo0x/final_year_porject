# main.py - AI Focus Monitor (Extended + Voice Assistant)

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
py_logging.getLogger("absl").setLevel(py_logging.ERROR) # Hide absl logs
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
# Config (try calibrated first, else default)
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
# Drawing / Utility
# -------------------------------------------------
def draw_badge(img, text, color, x=20, y=40):
    """Draw a status badge on the frame."""
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        cv2.rectangle(img, (x - 10, y - h - 10), (x + w + 10, y + 10), (255, 255, 255), -1)
        cv2.rectangle(img, (x - 10, y - h - 10), (x + w + 10, y + 10), (0, 0, 0), 2)
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    except Exception as e:
        logger.error(f"Error drawing badge: {e}")


def play_alert_sound(frequency=880, duration=0.3, volume=0.5):
    """Play a simple beep alert."""
    if not HAVE_SOUND:
        return

    try:
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        amplitude = np.iinfo(np.int16).max * volume
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        sd.play(wave.astype(np.int16), sample_rate)
    except Exception as e:
        logger.error(f"Error playing alert sound: {e}")


def speak(text: str):
    """Speak a short message using TTS."""
    if not HAVE_TTS or TTS_ENGINE is None:
        return

    try:
        TTS_ENGINE.stop()
        TTS_ENGINE.say(text)
        TTS_ENGINE.runAndWait()
    except Exception as e:
        logger.error(f"Error in voice assistant: {e}")


def validate_config():
    """Basic sanity checks on config values."""
    if cfg.EAR_SLEEPY_THRESHOLD <= cfg.EAR_BLINK_THRESHOLD:
        raise ValueError("EAR_SLEEPY_THRESHOLD must be greater than EAR_BLINK_THRESHOLD")

    if cfg.SLEEPY_FRAME_COUNT <= 0 or cfg.DISTRACT_FRAME_COUNT <= 0:
        raise ValueError("Frame count thresholds must be positive")

    if cfg.MAR_YAWN_THRESHOLD <= 0:
        raise ValueError("MAR_YAWN_THRESHOLD must be positive")


def initialize_camera(cam_index=0, max_retries=3):
    """Initialize camera with retries."""
    for attempt in range(max_retries):
        try:
            cap = cv2.VideoCapture(cam_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)

                ret, _ = cap.read()
                if ret:
                    logger.info(f"Camera {cam_index} initialized successfully")
                    return cap

                cap.release()

            logger.warning(f"Camera init attempt {attempt + 1} failed")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")

    raise RuntimeError(f"Failed to initialize camera {cam_index} after {max_retries} attempts")


# -------------------------------------------------
# Metrics processing
# -------------------------------------------------
def process_frame_metrics(results, frame_shape, ear_buffer):
    """Process frame and extract metrics."""
    h, w = frame_shape[:2]
    metrics = {'ear': 0.0, 'mar': 0.0, 'yaw': 1.0, 'pitch': 1.0}

    if not results or not results.multi_face_landmarks:
        return metrics, False

    try:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        raw_ear = (left_ear + right_ear) / 2.0

        ear_buffer.add(raw_ear)
        metrics['ear'] = ear_buffer.get_average()

        metrics['mar'] = mouth_aspect_ratio(landmarks, w, h)
        metrics['yaw'] = compute_yaw_proxy(landmarks, w, h)
        metrics['pitch'] = compute_pitch_proxy(landmarks, w, h)

        return metrics, True

    except Exception as e:
        logger.error(f"Error processing frame metrics: {e}")
        return metrics, False


# -------------------------------------------------
# Focus state classification
# -------------------------------------------------
def determine_focus_state(metrics, counters, cfg, face_detected):
    """Determine current focus state based on metrics (with No Face fix)."""

    if not face_detected:
        counters['noface'] += 1
        counters['sleepy'] = 0
        counters['distracted'] = 0
        return "No Face"

    counters['noface'] = 0

    is_distracted = (
        metrics['yaw'] > cfg.YAW_DISTRACT_THRESHOLD_RATIO or
        metrics['yaw'] < 1.0 / cfg.YAW_DISTRACT_THRESHOLD_RATIO or
        metrics['pitch'] > cfg.PITCH_DISTRACT_THRESHOLD_RATIO_HIGH or
        metrics['pitch'] < cfg.PITCH_DISTRACT_THRESHOLD_RATIO_LOW
    )

    if is_distracted:
        counters['distracted'] += 1
        counters['sleepy'] = 0
    else:
        counters['distracted'] = 0

    if cfg.EAR_BLINK_THRESHOLD < metrics['ear'] < cfg.EAR_SLEEPY_THRESHOLD:
        counters['sleepy'] += 1
    else:
        counters['sleepy'] = 0

    if metrics['mar'] > cfg.MAR_YAWN_THRESHOLD:
        return "Yawning"
    elif 0 < metrics['ear'] < cfg.EAR_BLINK_THRESHOLD:
        return "Blinking"
    elif counters['distracted'] > cfg.DISTRACT_FRAME_COUNT:
        return "Distracted"
    elif counters['sleepy'] > cfg.SLEEPY_FRAME_COUNT:
        return "Sleepy"
    else:
        return "Focused"


# -------------------------------------------------
# Overlay
# -------------------------------------------------
def draw_overlay_info(
    frame,
    state,
    metrics,
    score,
    colors,
    blink_count=0,
    yawn_count=0,
    focus_streak_sec=0.0,
    total_focus_sec=0.0,
):
    """Draw status, metrics and counters on frame."""
    try:
        draw_badge(frame, state, colors.get(state, (255, 0, 255)))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 0, 255)

        info_lines = [
            f"Score: {score:.1f}%",
            f"EAR: {metrics['ear']:.3f}",
            f"MAR: {metrics['mar']:.3f}",
            f"Yaw: {metrics['yaw']:.2f}",
            f"Pitch: {metrics['pitch']:.2f}",
            f"Blinks: {blink_count}",
            f"Yawns: {yawn_count}",
        ]

        if focus_streak_sec > 0:
            info_lines.append(f"Focus Streak: {focus_streak_sec/60:.1f} min")

        if total_focus_sec > 0:
            info_lines.append(f"Total Focus: {total_focus_sec/60:.1f} min")

        for i, line in enumerate(info_lines):
            y_pos = 70 + i * 25
            cv2.putText(frame, line, (20, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)

    except Exception as e:
        logger.error(f"Error drawing overlay: {e}")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    """Main application loop."""
    try:
        validate_config()
        logger.info("Configuration validated")

        cap = initialize_camera(cfg.CAM_INDEX)

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Session ID: {session_id}")

        log_interval = getattr(cfg, "LOG_INTERVAL", 0.5)  # seconds
        break_threshold = getattr(cfg, "FOCUS_BREAK_TIME_MIN", 45) * 60  # seconds
        voice_cooldown = getattr(cfg, "VOICE_COOLDOWN_SEC", 10)  # seconds between voice messages

        state = "No Face"
        prev_state = state
        start_time = time.time()
        prev_time = start_time
        last_log_time = start_time

        total_frames = 0
        focused_frames = 0

        blink_count = 0
        yawn_count = 0

        counters = {'sleepy': 0, 'distracted': 0, 'noface': 0}
        ear_buffer = EARBuffer(window_size=cfg.EAR_SMOOTH_WINDOW)

        last_focus_start = None
        focus_streak_sec = 0.0

        state_time = {
            "Focused": 0.0,
            "Distracted": 0.0,
            "Sleepy": 0.0,
            "Blinking": 0.0,
            "Yawning": 0.0,
            "No Face": 0.0,
        }

        alert_timer = None
        alert_playing = False

        # Voice assistant state
        last_voice_time = start_time - voice_cooldown
        last_voice_state = None

        colors = {
            "Focused": (0, 200, 0),
            "Distracted": (0, 165, 255),
            "Sleepy": (0, 0, 255),
            "No Face": (128, 128, 128),
            "Blinking": (255, 102, 0),
            "Yawning": (128, 0, 128),
        }

        csv_filename = f"session_log_{session_id}.csv"
        csv_path = Path(csv_filename)

        with csv_path.open('w', newline='', encoding='utf-8') as f, face_mesh_detector() as face_mesh:
            writer = csv.writer(f)
            writer.writerow(['t_sec', 'status', 'ear', 'mar', 'yaw_ratio', 'pitch_ratio'])

            logger.info("Monitoring started. Press 'q' or ESC to quit.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue

                frame = cv2.flip(frame, 1)
                total_frames += 1

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                metrics, face_detected = process_frame_metrics(results, frame.shape, ear_buffer)

                state = determine_focus_state(metrics, counters, cfg, face_detected)

                # Blink / yawn counters (on transitions)
                if state == "Blinking" and prev_state != "Blinking":
                    blink_count += 1
                if state == "Yawning" and prev_state != "Yawning":
                    yawn_count += 1
                prev_state = state

                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time

                # Time in each state
                if state in state_time:
                    state_time[state] += dt

                # Focus streak tracking
                if state == "Focused":
                    if last_focus_start is None:
                        last_focus_start = current_time
                    focused_frames += 1
                    focus_streak_sec = current_time - last_focus_start
                else:
                    last_focus_start = None
                    focus_streak_sec = 0.0

                # Beep alerts for sleepy / distracted
                if state in ["Sleepy", "Distracted"]:
                    if alert_timer is None:
                        alert_timer = current_time
                    elif not alert_playing and (current_time - alert_timer) > cfg.AUDIO_ALERT_TIME:
                        play_alert_sound()
                        alert_playing = True
                else:
                    alert_timer = None
                    alert_playing = False

                # Voice assistant (cooldown + state-based messages)
                if HAVE_TTS:
                    if state in ["Sleepy", "Distracted", "No Face"]:
                        if (current_time - last_voice_time) > voice_cooldown and state != last_voice_state:
                            if state == "Sleepy":
                                speak("You look sleepy. Take a short break and then refocus.")
                            elif state == "Distracted":
                                speak("Stay focused. Eyes on your work.")
                            elif state == "No Face":
                                speak("Face not visible. Please sit properly in front of the screen.")
                            last_voice_time = current_time
                            last_voice_state = state
                    else:
                        last_voice_state = None

                # Long continuous focus reminder (visual)
                if focus_streak_sec > break_threshold:
                    draw_badge(frame, "Long focus session. Consider a short break.", (0, 0, 255), x=20, y=260)

                focus_score = (focused_frames / total_frames * 100) if total_frames > 0 else 0.0

                draw_overlay_info(
                    frame,
                    state,
                    metrics,
                    focus_score,
                    colors,
                    blink_count=blink_count,
                    yawn_count=yawn_count,
                    focus_streak_sec=focus_streak_sec,
                    total_focus_sec=state_time["Focused"],
                )

                # CSV logging (reduced frequency)
                elapsed_time = current_time - start_time
                if elapsed_time - last_log_time >= log_interval:
                    writer.writerow([
                        round(elapsed_time, 2),
                        state,
                        round(metrics['ear'], 4),
                        round(metrics['mar'], 3),
                        round(metrics['yaw'], 3),
                        round(metrics['pitch'], 3),
                    ])
                    last_log_time = elapsed_time

                cv2.imshow("AI Focus Monitor", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

        # Session summary
        session_duration = time.time() - start_time
        overall_focus = (focused_frames / total_frames * 100) if total_frames > 0 else 0.0

        print("\n===== SESSION SUMMARY =====")
        print(f"Session ID     : {session_id}")
        print(f"Duration       : {session_duration/60:.1f} min")
        print(f"Total frames   : {total_frames}")
        print(f"Focus score    : {overall_focus:.1f}%")
        print(f"Total blinks   : {blink_count}")
        print(f"Total yawns    : {yawn_count}")
        print("\nTime spent in each state:")
        for s, t_sec in state_time.items():
            print(f"  {s:<10}: {t_sec/60:.2f} min")
        print("===========================\n")

    except KeyboardInterrupt:
        logger.info("Session interrupted by user")
    except Exception as e:
        logger.error(f"Error during session: {e}")
        raise
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Report
    try:
        report_png = f"focus_report_{session_id}.png"
        make_report(csv_filename, out_png=report_png)
        logger.info(f"Report saved to {report_png}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")


if __name__ == '__main__':
    main()