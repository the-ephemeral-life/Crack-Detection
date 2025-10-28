import cv2
from ultralytics import YOLO
from pathlib import Path

# --- 1. DEFINE PATHS ---
# Path to the folder where your runs are saved
RUNS_DIR = Path(__file__).resolve().parent / 'runs'

# Path to your specific training run
RUN_NAME = 'crack_detector_run_1'
BEST_MODEL_PATH = RUNS_DIR / RUN_NAME / 'weights' / 'best.pt'

# --- 2. LOAD THE TRAINED MODEL ---
# Check if the model file exists
if not BEST_MODEL_PATH.is_file():
    print(f"ERROR: Model file not found at {BEST_MODEL_PATH}")
    print("Please make sure the RUN_NAME is correct and the model has been trained.")
    exit()

# Load your custom-trained YOLOv8 model
try:
    model = YOLO(BEST_MODEL_PATH)
    print(f"✅ Model loaded successfully from {BEST_MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Failed to load model. {e}")
    exit()

# --- 3. INITIALIZE WEBCAM ---
# '0' is usually the default built-in webcam.
# If you have multiple cameras, you might need to try '1', '2', etc.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("🚀 Starting real-time crack detection... Press 'q' to quit.")

# --- 4. REAL-TIME DETECTION LOOP ---
while True:
    # Read one frame from the webcam
    ret, frame = cap.read()
    
    # If the frame was not read correctly, break the loop
    if not ret:
        print("ERROR: Failed to grab frame.")
        break

    # --- Run Prediction ---
    # We run the model on the current 'frame'
    # 'verbose=False' silences the text output for each frame
    results = model(frame, verbose=False)
    
    # --- Get Annotated Frame ---
    # 'results[0].plot()' returns the frame with bounding boxes
    # and labels drawn on it.
    annotated_frame = results[0].plot()

    # --- Display the Frame ---
    # Show the frame with detections in a window
    cv2.imshow("Real-Time Crack Detection", annotated_frame)

    # --- Exit Condition ---
    # Wait for 1 millisecond. If the 'q' key is pressed,
    # break the loop and end the program.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. CLEANUP ---
print("🛑 Stopping detection.")
cap.release()
cv2.destroyAllWindows()