import cv2
import datetime
from pathlib import Path
from ultralytics import YOLO
from picamera2 import Picamera2 # Import the new library

# --- 1. SETUP PATHS AND LOGGING ---
# Define paths
SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / 'runs'
RUN_NAME = 'crack_detector_run_1'
BEST_MODEL_PATH = RUNS_DIR / 'detect' / RUN_NAME / 'weights' / 'best.pt'

# Create a directory to store detection logs and images
SAVE_DIR = SCRIPT_DIR / 'crack_detections'
SAVE_DIR.mkdir(exist_ok=True) # Create the folder if it doesn't exist
LOG_FILE = SAVE_DIR / 'crack_log.txt'

# --- 2. LOAD THE TRAINED MODEL ---
if not BEST_MODEL_PATH.is_file():
    print(f"ERROR: Model file not found at {BEST_MODEL_PATH}")
    exit()

try:
    model = YOLO(BEST_MODEL_PATH)
    print(f"✅ Model loaded successfully from {BEST_MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Failed to load model. {e}")
    exit()

# --- 3. INITIALIZE RASPBERRY PI CAMERA (Picamera2) ---
try:
    picam2 = Picamera2()
    # Configure the camera for preview and capture
    # Using a size close to the model's input (640) is efficient
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    print("✅ Raspberry Pi Camera initialized successfully.")
except Exception as e:
    print(f"ERROR: Failed to initialize Pi Camera. {e}")
    print("Have you run 'sudo raspi-config' and enabled the camera?")
    exit()

print("🚀 Starting real-time crack detection... Press 'q' to quit.")

# --- 4. REAL-TIME DETECTION LOOP ---
while True:
    # --- Capture Frame ---
    # picam2.capture_array() returns an RGB NumPy array
    frame_rgb = picam2.capture_array()
    
    # Convert RGB to BGR (OpenCV and YOLO expect BGR)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # --- Run Prediction ---
    # verbose=False silences the text output for each frame
    results = model(frame_bgr, verbose=False)

    # --- Get Annotated Frame ---
    annotated_frame = results[0].plot()

    # --- !! NEW: LOGGING !! ---
    # Check if any cracks were detected (OBB results)
    if results[0].obb is not None and len(results[0].obb.xyxyxyxy) > 0:
        # Get current time for a unique filename
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f") # YearMonthDay_HourMinSec_Microsec
        
        # 1. Save the annotated image
        image_save_path = SAVE_DIR / f"crack_{timestamp_str}.jpg"
        cv2.imwrite(str(image_save_path), annotated_frame)
        
        # 2. Log the coordinates to the text file
        # Get the 4-point coordinates for all detected OBBs
        all_coords = results[0].obb.xyxyxyxy.cpu().numpy()
        
        with open(LOG_FILE, "a") as f:
            f.write(f"Timestamp: {timestamp.isoformat()}\n")
            f.write(f"Saved image: {image_save_path.name}\n")
            f.write(f"Detected {len(all_coords)} cracks:\n")
            for i, coords in enumerate(all_coords):
                f.write(f"  Crack {i+1} coordinates: {coords.tolist()}\n")
            f.write("-" * 20 + "\n") # Separator

    # --- Display the Frame ---
    cv2.imshow("Real-Time Crack Detection (Pi Camera)", annotated_frame)

    # --- Exit Condition ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. CLEANUP ---
print("🛑 Stopping detection.")
picam2.stop() # Stop the camera
cv2.destroyAllWindows()