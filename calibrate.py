#Optimized Calibration System
import time
import logging
from pathlib import Path
import cv2
import numpy as np
import focus_config as cfg
from focus_utils import *

logger = logging.getLogger(__name__)

class CalibrationSession:
    """Manages a calibration session with improved data collection and analysis"""
    
    def __init__(self, duration_sec=30):
        self.duration_sec = duration_sec
        self.neutral_duration = duration_sec * 0.4  # 40% for neutral position
        self.movement_duration = duration_sec * 0.6  # 60% for movements
        
        # Data storage
        self.all_ears = []
        self.neutral_data = {'yaws': [], 'pitches': [], 'ears': []}
        self.movement_data = {'yaws': [], 'pitches': [], 'ears': []}
        
        # Calibration state
        self.start_time = None
        self.phase = "neutral"  # "neutral" or "movement"
        
    def get_current_instruction(self, elapsed_time):
        """Get instruction text based on current calibration phase"""
        if elapsed_time < self.neutral_duration:
            remaining = int(self.neutral_duration - elapsed_time)
            return f"Look STRAIGHT at camera. Blink normally. ({remaining}s remaining)"
        else:
            remaining = int(self.duration_sec - elapsed_time)
            return f"Look around: UP, DOWN, LEFT, RIGHT, move head. ({remaining}s remaining)"
    
    def add_data_point(self, ear, yaw, pitch, elapsed_time):
        """Add a data point to the appropriate collection"""
        self.all_ears.append(ear)
        
        if elapsed_time < self.neutral_duration:
            self.neutral_data['ears'].append(ear)
            self.neutral_data['yaws'].append(yaw)
            self.neutral_data['pitches'].append(pitch)
        else:
            self.movement_data['ears'].append(ear)
            self.movement_data['yaws'].append(yaw)
            self.movement_data['pitches'].append(pitch)
    
    def is_complete(self, elapsed_time):
        """Check if calibration is complete"""
        return elapsed_time >= self.duration_sec
    
    def get_progress(self, elapsed_time):
        """Get calibration progress (0.0 to 1.0)"""
        return min(elapsed_time / self.duration_sec, 1.0)

def initialize_camera_for_calibration(cam_index=0):
    """Initialize camera with optimal settings for calibration"""
    try:
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam_index}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time processing
        
        # Test frame capture
        ret, _ = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Camera opened but cannot capture frames")
        
        logger.info(f"Camera {cam_index} initialized for calibration")
        return cap
        
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        raise

def draw_calibration_ui(frame, instruction, progress, phase_info=""):
    """Draw calibration UI elements on frame"""
    h, w = frame.shape[:2]
    
    try:
        # Draw semi-transparent overlay for instructions
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw instruction text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Main instruction
        text_size = cv2.getTextSize(instruction, font, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, instruction, (text_x, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Phase information
        if phase_info:
            phase_size = cv2.getTextSize(phase_info, font, 0.5, 1)[0]
            phase_x = (w - phase_size[0]) // 2
            cv2.putText(frame, phase_info, (phase_x, 60), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Progress bar
        bar_width = w - 40
        bar_height = 20
        bar_x = 20
        bar_y = 80
        
        # Progress bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Progress bar fill
        fill_width = int(progress * bar_width)
        color = (0, 255, 0) if progress < 1.0 else (0, 255, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Progress percentage
        progress_text = f"{progress * 100:.1f}%"
        progress_size = cv2.getTextSize(progress_text, font, 0.5, 1)[0]
        progress_x = (w - progress_size[0]) // 2
        cv2.putText(frame, progress_text, (progress_x, bar_y + 15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw face detection indicator
        cv2.circle(frame, (w - 30, 30), 10, (0, 255, 0), -1)  # Green circle for face detected
        cv2.putText(frame, "Face OK", (w - 80, 35), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        
    except Exception as e:
        logger.error(f"Error drawing calibration UI: {e}")

def analyze_calibration_data(session):
    """Analyze collected calibration data and compute thresholds"""
    try:
        if not session.all_ears:
            raise ValueError("No data collected during calibration")
        
        # Convert to numpy arrays for analysis
        all_ears = np.array(session.all_ears)
        neutral_yaws = np.array(session.neutral_data['yaws'])
        neutral_pitches = np.array(session.neutral_data['pitches'])
        movement_yaws = np.array(session.movement_data['yaws'])
        movement_pitches = np.array(session.movement_data['pitches'])
        
        logger.info(f"Analyzing {len(all_ears)} EAR samples")
        logger.info(f"Neutral phase: {len(neutral_yaws)} samples")
        logger.info(f"Movement phase: {len(movement_yaws)} samples")
        
        # EAR Analysis - Use more sophisticated percentile-based approach
        ear_stats = {
            'mean': np.mean(all_ears),
            'std': np.std(all_ears),
            'min': np.min(all_ears),
            'max': np.max(all_ears),
            'p5': np.percentile(all_ears, 5),
            'p25': np.percentile(all_ears, 25),
            'p75': np.percentile(all_ears, 75),
            'p95': np.percentile(all_ears, 95)
        }
        
        # Calculate EAR thresholds
        # Blink threshold: 5th percentile (lowest 5% of values are likely blinks)
        suggested_ear_blink = max(0.02, ear_stats['p5'])
        
        # Sleepy threshold: 25th percentile (drowsy eyes are partially closed)
        suggested_ear_sleepy = max(suggested_ear_blink + 0.02, ear_stats['p25'])
        
        # Head pose analysis
        if len(neutral_yaws) > 10 and len(movement_yaws) > 10:
            yaw_analysis = analyze_head_pose(neutral_yaws, movement_yaws, "yaw")
            pitch_analysis = analyze_head_pose(neutral_pitches, movement_pitches, "pitch")
        else:
            logger.warning("Insufficient data for head pose analysis. Using default values.")
            yaw_analysis = {'threshold_ratio': 2.5}
            pitch_analysis = {'threshold_high': 1.2, 'threshold_low': 0.8}
        
        return {
            'ear_blink': suggested_ear_blink,
            'ear_sleepy': suggested_ear_sleepy,
            'yaw_distract_ratio': yaw_analysis['threshold_ratio'],
            'pitch_distract_high': pitch_analysis['threshold_high'],
            'pitch_distract_low': pitch_analysis['threshold_low'],
            'ear_stats': ear_stats,
            'quality_score': calculate_calibration_quality(session)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing calibration data: {e}")
        return None

def analyze_head_pose(neutral_values, movement_values, pose_type):
    """Analyze head pose data to determine thresholds"""
    try:
        neutral_mean = np.mean(neutral_values)
        neutral_std = np.std(neutral_values)
        
        # Calculate deviations during movement phase
        movement_deviations = np.abs(movement_values - neutral_mean)
        max_deviation = np.percentile(movement_deviations, 85)  # 85th percentile for robustness
        
        if pose_type == "yaw":
            # For yaw, we use a ratio-based threshold
            threshold_ratio = neutral_mean + (max_deviation * 0.7)  # 70% of max observed deviation
            return {'threshold_ratio': max(1.5, min(5.0, threshold_ratio))}
        
        elif pose_type == "pitch":
            # For pitch, we calculate both high and low thresholds
            threshold_high = neutral_mean + (max_deviation * 0.7)
            threshold_low = neutral_mean - (max_deviation * 0.7)
            
            return {
                'threshold_high': max(0.5, min(2.0, threshold_high)),
                'threshold_low': max(0.2, min(1.0, threshold_low))
            }
            
    except Exception as e:
        logger.error(f"Error analyzing {pose_type} data: {e}")
        return {'threshold_ratio': 2.5} if pose_type == "yaw" else {'threshold_high': 1.2, 'threshold_low': 0.8}

def calculate_calibration_quality(session):
    """Calculate a quality score for the calibration session"""
    try:
        score = 100  # Start with perfect score
        
        # Penalize for insufficient data
        min_neutral_samples = session.neutral_duration * 15  # Expect ~15 FPS minimum
        min_movement_samples = session.movement_duration * 15
        
        if len(session.neutral_data['ears']) < min_neutral_samples:
            score -= 20
            
        if len(session.movement_data['ears']) < min_movement_samples:
            score -= 20
        
        # Check for reasonable variation in movement data
        if len(session.movement_data['yaws']) > 0:
            yaw_variation = np.std(session.movement_data['yaws'])
            if yaw_variation < 0.1:  # Very little head movement
                score -= 15
        
        # Check EAR data quality
        if len(session.all_ears) > 0:
            ear_variation = np.std(session.all_ears)
            if ear_variation < 0.01:  # Very little variation (possible camera issues)
                score -= 10
        
        return max(0, score)
        
    except Exception as e:
        logger.error(f"Error calculating calibration quality: {e}")
        return 50  # Return medium quality on error

def save_calibration_results(results, config_path="focus_config_calibrated.py"):
    """Save calibration results to a new config file"""
    try:
        config_content = f"""# focus_config.py - Calibrated Configuration
# Generated by calibration system on {time.strftime('%Y-%m-%d %H:%M:%S')}

# --- Core Detection Thresholds (CALIBRATED) ---
# Quality Score: {results['quality_score']}/100

# Eye Aspect Ratio (EAR) thresholds
EAR_BLINK_THRESHOLD = {results['ear_blink']:.3f}
EAR_SLEEPY_THRESHOLD = {results['ear_sleepy']:.3f}

# Head pose thresholds
YAW_DISTRACT_THRESHOLD_RATIO = {results['yaw_distract_ratio']:.2f}
PITCH_DISTRACT_THRESHOLD_RATIO_HIGH = {results['pitch_distract_high']:.2f}
PITCH_DISTRACT_THRESHOLD_RATIO_LOW = {results['pitch_distract_low']:.2f}

# Frame count thresholds
SLEEPY_FRAME_COUNT = 25
DISTRACT_FRAME_COUNT = 20

# Mouth Aspect Ratio (MAR) threshold for yawn detection
MAR_YAWN_THRESHOLD = 0.65

# --- Technical Parameters ---
NOFACE_TOLERANCE_FRAMES = 15
EAR_SMOOTH_WINDOW = 3
CAM_INDEX = 0
AUDIO_ALERT_TIME = 4.0

# --- Calibration Statistics ---
# EAR Statistics: Mean={results['ear_stats']['mean']:.3f}, Std={results['ear_stats']['std']:.3f}
# EAR Range: {results['ear_stats']['min']:.3f} to {results['ear_stats']['max']:.3f}
"""
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"Calibrated configuration saved to: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving calibration results: {e}")
        return False

def run_calibration(duration_sec=30, save_config=True):
    """
    Run the complete calibration process
    
    Args:
        duration_sec: Total calibration duration
        save_config: Whether to save calibrated config to file
    """
    try:
        logger.info(f"Starting {duration_sec}s calibration session...")
        
        # Initialize camera
        cap = initialize_camera_for_calibration(cfg.CAM_INDEX)
        
        # Create calibration session
        session = CalibrationSession(duration_sec)
        session.start_time = time.time()
        
        # Setup face mesh
        with face_mesh_detector() as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue
                
                # Calculate elapsed time
                elapsed_time = time.time() - session.start_time
                
                # Check if calibration is complete
                if session.is_complete(elapsed_time):
                    break
                
                # Process frame
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Detect face
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Calculate metrics
                    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    yaw = compute_yaw_proxy(landmarks, w, h)
                    pitch = compute_pitch_proxy(landmarks, w, h)
                    
                    # Add data point
                    session.add_data_point(avg_ear, yaw, pitch, elapsed_time)
                
                # Draw UI
                instruction = session.get_current_instruction(elapsed_time)
                progress = session.get_progress(elapsed_time)
                phase = "Neutral Position" if elapsed_time < session.neutral_duration else "Head Movement"
                
                draw_calibration_ui(frame, instruction, progress, f"Phase: {phase}")
                
                # Show frame
                cv2.imshow("Calibration - Press 'q' to quit early", frame)
                
                # Check for early exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("Calibration stopped by user")
                    break
        
        # Cleanup camera
        cap.release()
        cv2.destroyAllWindows()
        
        # Analyze results
        logger.info("Analyzing calibration data...")
        results = analyze_calibration_data(session)
        
        if results is None:
            logger.error("Calibration analysis failed")
            return False
        
        # Display results
        print("\n" + "="*60)
        print("🎯 CALIBRATION COMPLETE!")
        print("="*60)
        print(f"Quality Score: {results['quality_score']}/100")
        print(f"Data Points Collected: {len(session.all_ears)}")
        print("\nRecommended Configuration:")
        print("-" * 40)
        print(f"EAR_BLINK_THRESHOLD = {results['ear_blink']:.3f}")
        print(f"EAR_SLEEPY_THRESHOLD = {results['ear_sleepy']:.3f}")
        print(f"YAW_DISTRACT_THRESHOLD_RATIO = {results['yaw_distract_ratio']:.2f}")
        print(f"PITCH_DISTRACT_THRESHOLD_RATIO_HIGH = {results['pitch_distract_high']:.2f}")
        print(f"PITCH_DISTRACT_THRESHOLD_RATIO_LOW = {results['pitch_distract_low']:.2f}")
        print("-" * 40)
        
        # Quality assessment
        if results['quality_score'] >= 80:
            print("✅ Excellent calibration quality!")
        elif results['quality_score'] >= 60:
            print("⚠️  Good calibration quality.")
        else:
            print("❌ Poor calibration quality. Consider recalibrating.")
            print("   Tips: Ensure good lighting, look around more during movement phase.")
        
        # Save configuration if requested
        if save_config:
            if save_calibration_results(results):
                print(f"\n📄 Calibrated configuration saved to 'focus_config_calibrated.py'")
                print("   Replace your current focus_config.py with this file to use the new settings.")
            else:
                print("\n❌ Failed to save calibrated configuration")
        
        print("\n" + "="*60)
        return True
        
    except KeyboardInterrupt:
        logger.info("Calibration interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return False
    finally:
        # Ensure cleanup
        try:
            cv2.destroyAllWindows()
        except:
            pass

def quick_calibration(duration_sec=15):
    """Run a quick 15-second calibration for basic setup"""
    logger.info("Running quick calibration...")
    return run_calibration(duration_sec, save_config=False)

def validate_current_config():
    """Validate current configuration against recommended ranges"""
    try:
        print("\n🔍 VALIDATING CURRENT CONFIGURATION")
        print("="*50)
        
        # Check EAR thresholds
        if cfg.EAR_SLEEPY_THRESHOLD <= cfg.EAR_BLINK_THRESHOLD:
            print("❌ EAR_SLEEPY_THRESHOLD must be greater than EAR_BLINK_THRESHOLD")
        else:
            print("✅ EAR thresholds are properly ordered")
        
        # Check reasonable ranges
        issues = []
        if cfg.EAR_BLINK_THRESHOLD < 0.02 or cfg.EAR_BLINK_THRESHOLD > 0.15:
            issues.append("EAR_BLINK_THRESHOLD outside typical range (0.02-0.15)")
        
        if cfg.EAR_SLEEPY_THRESHOLD < 0.1 or cfg.EAR_SLEEPY_THRESHOLD > 0.4:
            issues.append("EAR_SLEEPY_THRESHOLD outside typical range (0.1-0.4)")
        
        if cfg.YAW_DISTRACT_THRESHOLD_RATIO < 1.5 or cfg.YAW_DISTRACT_THRESHOLD_RATIO > 5.0:
            issues.append("YAW_DISTRACT_THRESHOLD_RATIO outside typical range (1.5-5.0)")
        
        if issues:
            print("\n⚠️  Potential Issues:")
            for issue in issues:
                print(f"   • {issue}")
            print("\nConsider running calibration to optimize these values.")
        else:
            print("✅ All configuration values appear reasonable")
        
        return len(issues) == 0
        
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False

def interactive_calibration_menu():
    """Interactive menu for calibration options"""
    while True:
        print("\n" + "="*50)
        print("🎯 AI FOCUS MONITOR - CALIBRATION MENU")
        print("="*50)
        print("1. Full Calibration (30 seconds) - Recommended")
        print("2. Quick Calibration (15 seconds)")
        print("3. Validate Current Configuration")
        print("4. View Current Configuration")
        print("5. Exit")
        print("-" * 50)
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                if run_calibration(30, save_config=True):
                    print("\n✅ Full calibration completed successfully!")
                else:
                    print("\n❌ Full calibration failed.")
                    
            elif choice == "2":
                if quick_calibration(15):
                    print("\n✅ Quick calibration completed!")
                    save_choice = input("\nSave configuration? (y/n): ").lower().strip()
                    if save_choice == 'y':
                        # Would need to implement saving for quick calibration
                        print("Quick calibration results not saved. Run full calibration to save.")
                else:
                    print("\n❌ Quick calibration failed.")
                    
            elif choice == "3":
                validate_current_config()
                
            elif choice == "4":
                print_current_config()
                
            elif choice == "5":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Menu error: {e}")
            print("An error occurred. Please try again.")

def print_current_config():
    """Display current configuration values"""
    print("\n📋 CURRENT CONFIGURATION")
    print("-" * 30)
    print(f"EAR_BLINK_THRESHOLD: {cfg.EAR_BLINK_THRESHOLD:.3f}")
    print(f"EAR_SLEEPY_THRESHOLD: {cfg.EAR_SLEEPY_THRESHOLD:.3f}")
    print(f"YAW_DISTRACT_THRESHOLD_RATIO: {cfg.YAW_DISTRACT_THRESHOLD_RATIO:.2f}")
    print(f"PITCH_DISTRACT_THRESHOLD_RATIO_HIGH: {cfg.PITCH_DISTRACT_THRESHOLD_RATIO_HIGH:.2f}")
    print(f"PITCH_DISTRACT_THRESHOLD_RATIO_LOW: {cfg.PITCH_DISTRACT_THRESHOLD_RATIO_LOW:.2f}")
    print(f"MAR_YAWN_THRESHOLD: {cfg.MAR_YAWN_THRESHOLD:.2f}")
    print(f"SLEEPY_FRAME_COUNT: {cfg.SLEEPY_FRAME_COUNT}")
    print(f"DISTRACT_FRAME_COUNT: {cfg.DISTRACT_FRAME_COUNT}")

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🎯 AI Focus Monitor - Calibration System")
    print("This will help you personalize the detection thresholds.")
    print("\nFor best results:")
    print("• Ensure good, even lighting")
    print("• Position camera at eye level")
    print("• Sit normally as you would during focus sessions")
    print("• Follow the on-screen instructions carefully")
    
    # Run interactive menu
    interactive_calibration_menu()