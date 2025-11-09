import cv2
import socket
import struct
import pickle
from picamera2 import Picamera2
from libcamera import controls
import time

# --- CONFIGURATION ---
LAPTOP_IP = '192.168.1.100'  # IMPORTANT: Change this to your Laptop's IP address
PORT = 9999
FRAME_RATE = 12
# ---------------------

# 1. INITIALIZE THE PI CAMERA
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={"FrameDurationLimits": (int(1e6/FRAME_RATE), int(1e6/FRAME_RATE))}
)
picam2.configure(config)
picam2.start()

# Give the camera a moment to warm up
time.sleep(1.0)
print(f"Camera started at {FRAME_RATE} FPS. Connecting to {LAPTOP_IP}...")

# 2. INITIALIZE THE NETWORK CONNECTION
try:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((LAPTOP_IP, PORT))
    print(f"✅ Connected to laptop at {LAPTOP_IP}:{PORT}")
except Exception as e:
    print(f"❌ FAILED TO CONNECT to {LAPTOP_IP}:{PORT}. {e}")
    print("Is the laptop_server.py script running on the laptop?")
    picam2.stop()
    exit()

# 3. START THE STREAMING LOOP
try:
    while True:
        # 1. Capture a frame
        frame = picam2.capture_array()
        
        # 2. Compress the frame to JPEG (much faster over network)
        # 90% quality is a good balance
        ret, frame_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        if not ret:
            print("Failed to encode frame")
            continue
            
        # 3. Serialize the data
        data = pickle.dumps(frame_data)
        
        # 4. Pack and send the frame
        # 'L' means unsigned long, for the size
        message = struct.pack("L", len(data)) + data
        client_socket.sendall(message)
        
        # This (optional) small delay helps keep the loop from
        # overwhelming the network buffer if the camera is *slightly*
        # faster than the requested FPS.
        time.sleep(0.01)

except (BrokenPipeError, ConnectionResetError):
    print("Lost connection to laptop.")
except KeyboardInterrupt:
    print("Stopping stream.")
finally:
    client_socket.close()
    picam2.stop()
    print("Connection closed. Pi script terminated.")