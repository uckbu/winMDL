import cv2
import os
import time
from collections import deque
from inference_sdk import InferenceHTTPClient

# --- Configuration ---
API_URL = "https://serverless.roboflow.com"
API_KEY = "zoipelF67Mn8RniiR4eZ"
MODEL_ID = "meteors-8m2qc-rruxg/1"

VIDEOS_DIR = "videos"       # where event videos will be stored
FPS = 30                    # expected frame rate
PRE_EVENT_FRAMES = 10       # frames to buffer before meteor appears
POST_EVENT_FRAMES = 20      # frames to capture after meteor disappears

# --- Motion Detection Configuration ---
MOTION_THRESHOLD_VALUE = 25      # threshold value for difference image
MOTION_PIXEL_THRESHOLD = 2000    # number of changed pixels required to trigger inference

# Create videos directory if needed
if not os.path.exists(VIDEOS_DIR):
    os.makedirs(VIDEOS_DIR)

# --- Initialize Inference Client ---
CLIENT = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
)

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Pre-event buffer to hold the last few frames
pre_event_buffer = deque(maxlen=PRE_EVENT_FRAMES)
# List to keep the frames for the current event (pre-event + event duration + post-event)
event_frames = []

meteor_event_active = False  # Flag indicating an ongoing meteor event
post_event_counter = 0       # Counts consecutive frames with no meteor detection

# Variable for motion detection: previous gray frame
prev_gray = None

def save_video(frames):
    """Saves a list of frames as a video file in the 'videos' directory."""
    if not frames:
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"meteor_{timestamp}.mp4"
    filepath = os.path.join(VIDEOS_DIR, filename)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter(filepath, fourcc, FPS, (width, height))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()
    print(f"Saved event video: {filepath}")

print("Starting webcam processing. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize frame to 640x640 as required
    frame_resized = cv2.resize(frame, (640, 640))
    
    # Convert to grayscale and blur to reduce noise for motion detection
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Determine if sufficient motion is detected by comparing with the previous frame
    movement_detected = False
    if prev_gray is not None:
        frame_delta = cv2.absdiff(gray_blurred, prev_gray)
        _, thresh = cv2.threshold(frame_delta, MOTION_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        changed_pixels = cv2.countNonZero(thresh)
        if changed_pixels > MOTION_PIXEL_THRESHOLD:
            movement_detected = True
    # Update the previous frame for the next iteration
    prev_gray = gray_blurred.copy()
    
    # Decide whether to run inference:
    # Always run inference if a meteor event is already active to check for event termination.
    # Otherwise, only run inference if motion is detected.
    run_inference = meteor_event_active or movement_detected
    
    meteor_detected = False  # Default to false unless proven by inference
    predictions = []
    
    if run_inference:
        temp_filename = "temp_frame.jpg"
        cv2.imwrite(temp_filename, frame_resized)
        try:
            result = CLIENT.infer(temp_filename, model_id=MODEL_ID)
            predictions = result.get("predictions", [])
            meteor_detected = len(predictions) > 0
        except Exception as e:
            print("Inference error:", e)
        # Remove the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    # Optionally, draw detection bounding boxes on the frame if a meteor is detected.
    display_frame = frame_resized.copy()
    if meteor_detected:
        for pred in predictions:
            # Assume pred["x"], pred["y"] are the center coordinates.
            cx = pred["x"]
            cy = pred["y"]
            w = pred["width"]
            h = pred["height"]
            # Compute top-left and bottom-right corners.
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Meteor: {pred['confidence']:.2f}", (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Always add the current display frame to the pre-event buffer
    pre_event_buffer.append(display_frame.copy())

    # Meteor event handling:
    if meteor_event_active:
        event_frames.append(display_frame.copy())
        if meteor_detected:
            post_event_counter = 0  # reset the counter if meteor is still visible
        else:
            post_event_counter += 1

        # When POST_EVENT_FRAMES consecutive frames have no meteor, conclude the event
        if post_event_counter >= POST_EVENT_FRAMES:
            save_video(event_frames)
            event_frames = []
            meteor_event_active = False
            post_event_counter = 0
    else:
        # If no event is active but a meteor is detected, begin a new event.
        if meteor_detected:
            meteor_event_active = True
            # Start the event frames with the pre-event buffer
            event_frames = list(pre_event_buffer)
            event_frames.append(display_frame.copy())
            post_event_counter = 0

    # Display the processed frame with detection overlay
    cv2.imshow("Webcam Meteor Detector", display_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
cap.release()
cv2.destroyAllWindows()
