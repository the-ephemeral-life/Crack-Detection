from ultralytics import YOLO
from pathlib import Path

# --- 1. DEFINE PATHS ---
# Path to the folder where your runs are saved
RUNS_DIR = Path(__file__).resolve().parents[1] / 'runs'

# Path to your specific training run (change this if your folder is named differently)
RUN_NAME = 'crack_detector_run_1'
BEST_MODEL_PATH = RUNS_DIR / RUN_NAME / 'weights' / 'best.pt'

# Path to the image you want to test
IMAGE_TO_TEST = 'C:/Users/arora/OneDrive/Desktop/WhatsApp Image 2025-10-08 at 19.36.46_a49a8214.jpg' # IMPORTANT: Change this to your image path!

# --- 2. LOAD THE TRAINED MODEL ---
# Check if the model file exists
if not BEST_MODEL_PATH.is_file():
    print(f"ERROR: Model file not found at {BEST_MODEL_PATH}")
    exit()

# Load your custom-trained YOLOv8 model
model = YOLO(BEST_MODEL_PATH)
print(f"✅ Model loaded successfully from {BEST_MODEL_PATH}")

# --- 3. RUN PREDICTION ---
# Run inference on the image
results = model(IMAGE_TO_TEST)
print("✅ Prediction complete.")

# --- 4. SHOW RESULTS ---
# The results object contains the detected boxes, masks, etc.
# The .show() method displays the image with the predictions drawn on it.
for r in results:
    r.show()

print("🖼️ Displaying image with detected cracks. Close the image window to exit.")