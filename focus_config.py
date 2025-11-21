# focus_config.py - FIXED Configuration (No Face Detection Bug Resolved)

# --- CORRECTED VALUES FOR BETTER DETECTION ---
EAR_BLINK_THRESHOLD = 0.087
EAR_SLEEPY_THRESHOLD = 0.260           # Increased from 0.227 (better sleepy detection)
YAW_DISTRACT_THRESHOLD_RATIO = 2.80    # Increased from 1.50 (less sensitive distraction)
PITCH_DISTRACT_THRESHOLD_RATIO_HIGH = 0.90  # Increased from 0.61 (less sensitive)
PITCH_DISTRACT_THRESHOLD_RATIO_LOW = 0.40   # Increased from 0.20 (less sensitive)

SLEEPY_FRAME_COUNT = 25
DISTRACT_FRAME_COUNT = 20

# Other values stay the same
MAR_YAWN_THRESHOLD = 0.65
NOFACE_TOLERANCE_FRAMES = 15  # This is now ignored - immediate "No Face" detection
EAR_SMOOTH_WINDOW = 3
CAM_INDEX = 0
AUDIO_ALERT_TIME = 4.0

# How often to log data to CSV (seconds)
LOG_INTERVAL = 0.5

# How many minutes of continuous focus before break reminder
FOCUS_BREAK_TIME_MIN = 45