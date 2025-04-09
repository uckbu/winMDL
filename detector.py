import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
import time
import datetime
import os
import platform

# === CONFIGURATION ===
MODEL_PATH = "yolov5s.onnx"   # Path to your ONNX model

# Webcam resolution settings (adjust if necessary)
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720  
# The network expects an input of this resolution
FRAME_WIDTH, FRAME_HEIGHT = 640, 640
FPS = 30
PRE_BUFFER_SIZE = 60   # 60 frames before event (approx. 2 sec)
POST_BUFFER_SIZE = 60  # 60 frames after event ends
DETECTION_THRESHOLD = 0.5  # Confidence threshold for meteor detection
VIDEOS_DIR = "videos"

# Ensure the output folder exists
os.makedirs(VIDEOS_DIR, exist_ok=True)

# === SET ONNXRUNTIME PROVIDERS BASED ON PLATFORM ===
if platform.system() == "Windows":
    # On Windows, TensorrtExecutionProvider is typically unavailable
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
else:
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name

def preprocess(frame):
    """
    Resize, convert to RGB, normalize, and reformat for ONNX.
    """
    img = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # from HWC to CHW
    img = np.expand_dims(img, axis=0)     # add batch dimension
    return img

def run_inference(frame):
    """
    Run the model on a preprocessed frame.
    Returns True if any detection has confidence above the threshold.
    """
    input_tensor = preprocess(frame)
    outputs = session.run(None, {input_name: input_tensor})
    # Assuming model outputs shape [1, N, 85] with index 4 as confidence
    detections = np.squeeze(outputs[0])
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)
    if detections.size and np.any(detections[:, 4] > DETECTION_THRESHOLD):
        return True
    return False

# === INITIALIZE CAMERA (For Windows with a standard webcam) ===
cap = cv2.VideoCapture(0)  # Use the default webcam device
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

# Buffers and state variables
pre_buffer = deque(maxlen=PRE_BUFFER_SIZE)
recording_frames = []  # Will hold frames for the current meteor event
recording = False
post_counter = 0

print("[INFO] Starting video capture and meteor detection...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Always add the current frame to the pre-buffer
        pre_buffer.append(frame.copy())

        # Run inference on the current frame
        meteor_detected = run_inference(frame)

        if meteor_detected:
            # Start a new event if not already recording
            if not recording:
                print("[EVENT] Meteor detected! Starting event recording.")
                # Prepend the pre-buffer (frames before the meteor appeared)
                recording_frames = list(pre_buffer)
                recording = True
            # Append the current frame and reset post-event counter
            recording_frames.append(frame.copy())
            post_counter = 0
        else:
            if recording:
                # Meteor no longer detected; keep recording post-event frames
                recording_frames.append(frame.copy())
                post_counter += 1
                # Once we have recorded enough post-event frames, finalize the video
                if post_counter >= POST_BUFFER_SIZE:
                    finish_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    video_filename = os.path.join(VIDEOS_DIR, f"{finish_time}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(video_filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
                    
                    for frm in recording_frames:
                        out.write(frm)
                    out.release()

                    print(f"[INFO] Saved event video: {video_filename}")
                    
                    # Reset recording state
                    recording = False
                    recording_frames = []
                    post_counter = 0

        # Display the frame; the window is now labeled "Webcam"
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

# Clean up
cap.release()
cv2.destroyAllWindows()
