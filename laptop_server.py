import cv2
import socket
import struct
import pickle
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
# Load your trained model
MODEL_PATH = 'runs/crack_detector_run_1/weights/best.pt'
HOST_IP = '0.0.0.0'  # Listen on all available network interfaces
PORT = 9999
# ---------------------

# 1. LOAD THE YOLO MODEL
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ FAILED TO LOAD MODEL from {MODEL_PATH}. {e}")
    exit()

# 2. SET UP THE SERVER SOCKET
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST_IP, PORT))
server_socket.listen(1)
print(f"🖥️ Server listening on {HOST_IP}:{PORT}...")
print("Waiting for Raspberry Pi to connect...")

conn, addr = server_socket.accept()
print(f"✅ Pi connected from: {addr}")

# 3. PREPARE TO RECEIVE DATA
data = b""
payload_size = struct.calcsize("L") # Size of the packed message length

try:
    while True:
        # 1. Receive data from the Pi until we have a complete "size" message
        while len(data) < payload_size:
            data += conn.recv(4096)
            if not data:
                raise ConnectionError("Pi disconnected.")
        
        # 2. Unpack the size of the frame
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]
        
        # 3. Receive data until we have the full frame
        while len(data) < msg_size:
            data += conn.recv(4096)
            if not data:
                raise ConnectionError("Pi disconnected.")

        # 4. Extract, de-serialize, and decompress the frame
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        frame_jpg = pickle.loads(frame_data)
        frame = cv2.imdecode(frame_jpg, cv2.IMREAD_COLOR)

        if frame is None:
            print("Received a bad frame, skipping...")
            continue
            
        # 5. RUN CRACK DETECTION
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # 6. Display the result
        cv2.imshow("Crack Detection (from Pi)", annotated_frame)
        
        # 7. Check for 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except (ConnectionResetError, ConnectionError, struct.error):
    print("❌ Connection lost with the Raspberry Pi.")
except KeyboardInterrupt:
    print("Stopping server.")
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("Server shut down.")