# focus_utils.py - Optimized Focus Detection Utilities
import math
import logging
from collections import deque
from typing import List, Tuple, Optional
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Facial landmark indices for different features
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_VERTICAL = [13, 14]
MOUTH_HORIZONTAL = [61, 291]

# Additional landmark indices for better head pose estimation
NOSE_TIP = 1
NOSE_BRIDGE = 6
CHIN = 152
LEFT_FACE = 234
RIGHT_FACE = 454
LEFT_EYE_CENTER = 133
RIGHT_EYE_CENTER = 362

def euclid_dist(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Calculate Euclidean distance between two points"""
    try:
        return float(np.linalg.norm(point_a - point_b))
    except Exception as e:
        logger.error(f"Error calculating Euclidean distance: {e}")
        return 0.0

def get_landmark_coordinates(landmarks, indices: List[int], image_w: int, image_h: int) -> np.ndarray:
    """Extract landmark coordinates and convert to pixel coordinates"""
    try:
        coords = []
        for idx in indices:
            landmark = landmarks[idx]
            x = landmark.x * image_w
            y = landmark.y * image_h
            coords.append([x, y])
        return np.array(coords, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error extracting landmark coordinates: {e}")
        return np.array([])

def eye_aspect_ratio(landmarks, eye_indices: List[int], image_w: int, image_h: int) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for drowsiness detection
    
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    where p1,p2,p3,p4,p5,p6 are the eye landmark points
    """
    try:
        if len(eye_indices) != 6:
            raise ValueError("Eye indices must contain exactly 6 points")
        
        points = get_landmark_coordinates(landmarks, eye_indices, image_w, image_h)
        if points.size == 0:
            return 0.0
        
        # Calculate vertical distances
        vertical_1 = euclid_dist(points[1], points[5])  # p2 to p6
        vertical_2 = euclid_dist(points[2], points[4])  # p3 to p5
        
        # Calculate horizontal distance
        horizontal = euclid_dist(points[0], points[3])  # p1 to p4
        
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return max(0.0, min(1.0, ear))  # Clamp between 0 and 1
        
    except Exception as e:
        logger.error(f"Error calculating eye aspect ratio: {e}")
        return 0.0

def mouth_aspect_ratio(landmarks, image_w: int, image_h: int) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) for yawn detection
    
    MAR = mouth_height / mouth_width
    """
    try:
        # Get mouth landmarks
        top_lip = landmarks[MOUTH_VERTICAL[0]]
        bottom_lip = landmarks[MOUTH_VERTICAL[1]]
        left_corner = landmarks[MOUTH_HORIZONTAL[0]]
        right_corner = landmarks[MOUTH_HORIZONTAL[1]]
        
        # Convert to pixel coordinates
        top_point = np.array([top_lip.x * image_w, top_lip.y * image_h])
        bottom_point = np.array([bottom_lip.x * image_w, bottom_lip.y * image_h])
        left_point = np.array([left_corner.x * image_w, left_corner.y * image_h])
        right_point = np.array([right_corner.x * image_w, right_corner.y * image_h])
        
        # Calculate mouth dimensions
        mouth_height = euclid_dist(top_point, bottom_point)
        mouth_width = euclid_dist(left_point, right_point)
        
        if mouth_width == 0:
            return 0.0
        
        mar = mouth_height / mouth_width
        return max(0.0, min(2.0, mar))  # Clamp to reasonable range
        
    except Exception as e:
        logger.error(f"Error calculating mouth aspect ratio: {e}")
        return 0.0

def compute_yaw_proxy(landmarks, image_w: int, image_h: int) -> float:
    """
    Compute head yaw (left-right rotation) proxy
    
    Uses the ratio of distances from nose tip to left/right face edges
    """
    try:
        nose_tip = landmarks[NOSE_TIP]
        left_face = landmarks[LEFT_FACE]
        right_face = landmarks[RIGHT_FACE]
        
        # Convert to pixel coordinates
        nose_point = np.array([nose_tip.x * image_w, nose_tip.y * image_h])
        left_point = np.array([left_face.x * image_w, left_face.y * image_h])
        right_point = np.array([right_face.x * image_w, right_face.y * image_h])
        
        # Calculate distances
        left_distance = euclid_dist(nose_point, left_point)
        right_distance = euclid_dist(nose_point, right_point)
        
        if right_distance == 0:
            return 1.0
        
        yaw_ratio = left_distance / right_distance
        return max(0.1, min(10.0, yaw_ratio))  # Clamp to reasonable range
        
    except Exception as e:
        logger.error(f"Error computing yaw proxy: {e}")
        return 1.0

def compute_pitch_proxy(landmarks, image_w: int, image_h: int) -> float:
    """
    Compute head pitch (up-down rotation) proxy
    
    Uses the ratio of eye-to-nose distance vs nose-to-chin distance
    """
    try:
        left_eye = landmarks[LEFT_EYE_CENTER]
        right_eye = landmarks[RIGHT_EYE_CENTER]
        nose_tip = landmarks[NOSE_TIP]
        chin = landmarks[CHIN]
        
        # Convert to pixel coordinates
        left_eye_point = np.array([left_eye.x * image_w, left_eye.y * image_h])
        right_eye_point = np.array([right_eye.x * image_w, right_eye.y * image_h])
        nose_point = np.array([nose_tip.x * image_w, nose_tip.y * image_h])
        chin_point = np.array([chin.x * image_w, chin.y * image_h])
        
        # Calculate eye midpoint
        eye_midpoint = (left_eye_point + right_eye_point) / 2.0
        
        # Calculate distances
        eye_to_nose_dist = abs(eye_midpoint[1] - nose_point[1])
        nose_to_chin_dist = abs(nose_point[1] - chin_point[1])
        
        if nose_to_chin_dist == 0:
            return 1.0
        
        pitch_ratio = eye_to_nose_dist / nose_to_chin_dist
        return max(0.1, min(5.0, pitch_ratio))  # Clamp to reasonable range
        
    except Exception as e:
        logger.error(f"Error computing pitch proxy: {e}")
        return 1.0

class EARBuffer:
    """
    Smoothing buffer for Eye Aspect Ratio to reduce noise
    """
    def __init__(self, window_size: int = 3):
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        self.buffer = deque(maxlen=window_size)
        self.window_size = window_size
    
    def add(self, value: float) -> None:
        """Add a new EAR value to the buffer"""
        if not isinstance(value, (int, float)):
            logger.warning(f"Invalid EAR value type: {type(value)}")
            return
        
        # Clamp value to reasonable range
        clamped_value = max(0.0, min(1.0, value))
        self.buffer.append(clamped_value)
    
    def get_average(self) -> float:
        """Get the smoothed average EAR value"""
        if not self.buffer:
            return 0.0
        return sum(self.buffer) / len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is at full capacity"""
        return len(self.buffer) == self.window_size
    
    def clear(self) -> None:
        """Clear the buffer"""
        self.buffer.clear()

def face_mesh_detector(**kwargs):
    """
    Create and configure MediaPipe Face Mesh detector
    
    Returns a configured FaceMesh instance with optimized settings
    """
    default_params = {
        'max_num_faces': 1,
        'refine_landmarks': True,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    }
    
    # Update with any user-provided parameters
    default_params.update(kwargs)
    
    try:
        return mp_face_mesh.FaceMesh(**default_params)
    except Exception as e:
        logger.error(f"Error creating face mesh detector: {e}")
        # Return with minimal confidence settings as fallback
        return mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

def validate_landmarks(landmarks, required_indices: List[int]) -> bool:
    """
    Validate that all required landmark indices exist and are valid
    """
    try:
        if not landmarks or len(landmarks) == 0:
            return False
        
        landmark_list = landmarks
        max_index = len(landmark_list)
        
        for idx in required_indices:
            if idx >= max_index or idx < 0:
                return False
            
            landmark = landmark_list[idx]
            if not hasattr(landmark, 'x') or not hasattr(landmark, 'y'):
                return False
            
            # Check for reasonable coordinate values
            if not (0 <= landmark.x <= 1) or not (0 <= landmark.y <= 1):
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating landmarks: {e}")
        return False

def compute_all_metrics(landmarks, image_w: int, image_h: int) -> dict:
    """
    Compute all face metrics at once for efficiency
    """
    metrics = {
        'ear_left': 0.0,
        'ear_right': 0.0,
        'ear_avg': 0.0,
        'mar': 0.0,
        'yaw': 1.0,
        'pitch': 1.0,
        'valid': False
    }
    
    try:
        # Validate required landmarks
        all_required = LEFT_EYE + RIGHT_EYE + MOUTH_VERTICAL + MOUTH_HORIZONTAL
        all_required += [NOSE_TIP, LEFT_FACE, RIGHT_FACE, LEFT_EYE_CENTER, RIGHT_EYE_CENTER, CHIN]
        
        if not validate_landmarks(landmarks, all_required):
            return metrics
        
        # Calculate eye aspect ratios
        metrics['ear_left'] = eye_aspect_ratio(landmarks, LEFT_EYE, image_w, image_h)
        metrics['ear_right'] = eye_aspect_ratio(landmarks, RIGHT_EYE, image_w, image_h)
        metrics['ear_avg'] = (metrics['ear_left'] + metrics['ear_right']) / 2.0
        
        # Calculate mouth aspect ratio
        metrics['mar'] = mouth_aspect_ratio(landmarks, image_w, image_h)
        
        # Calculate head pose proxies
        metrics['yaw'] = compute_yaw_proxy(landmarks, image_w, image_h)
        metrics['pitch'] = compute_pitch_proxy(landmarks, image_w, image_h)
        
        metrics['valid'] = True
        
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
    
    return metrics