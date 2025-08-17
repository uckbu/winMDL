# winMDL — Simple Meteor Detector (Webcam)

This project turns a webcam into a simple meteor detector. It watches the night sky through your camera, and when it “sees” something that looks like a meteor, it records a short video clip of the event and saves it in a `videos/` folder.

You can run it in two ways:
- Local model (no internet after setup): `detector.py` uses a YOLO model in ONNX format on your machine.
- Cloud model (internet required): `winMDL.py` sends frames to a hosted model API on Roboflow and gets predictions back.

No computer vision background needed. Follow the steps below and you’ll be set.

## What it does
- The app reads images from your webcam many times per second (these are called “frames”).
- It checks each frame for patterns that look like meteors.
- When a meteor is detected, it starts an “event”: it includes a few frames from before the meteor appeared and keeps recording a bit after it disappears, then saves the whole event as a video.
- You can stop the app any time by pressing the `q` key in the video window.

## What you need
- Laptop with a built-in or USB webcam
- Python 3.9+ installed
- Internet connection if you plan to use the cloud model (`winMDL.py`)

## Setup (Windows, zsh)
It’s best to use a virtual environment so your Python packages don’t conflict with other projects.

```bash
# From the project folder
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Core packages used by the two scripts
pip install opencv-python numpy onnxruntime inference-sdk
```

Grant camera access to the terminal or VS Code when prompted. If you don’t see a prompt:
- System Settings → Privacy & Security → Camera → enable your Terminal and/or Visual Studio Code.

## Option A: Run the local model (detector.py)
This uses an ONNX model file named `yolov5s.onnx` and runs entirely on your computer.

1) Get the model file (one-time):
- Train a YOLOv5 ONNX model (e.g., “yolov5s.onnx”) to detect meteors. Save it in the project folder and name it exactly `yolov5s.onnx`.
- If you don’t have a model handy, prefer Option B first (cloud model) which doesn’t require a local model file (and is already trained).

2) Run it:
```bash
# From the project folder, with your venv activated
python detector.py
```
What you’ll see:
- A window titled “Webcam”. Press `q` to quit.
- When a meteor-like object is detected, the app saves a clip in `videos/`.

Notes for macOS:
- If you see an error mentioning “ExecutionProvider” or “TensorRT/CUDA”: your Mac likely doesn’t have those GPU backends. The script already falls back to CPU, but if you still get errors, open `detector.py` and change the providers block to CPU only:
  ```python
  providers = ['CPUExecutionProvider']
  ```
  The app will still work—just on the CPU.

## Option B: Run the cloud model (winMDL.py)
This version sends a resized frame to a hosted model and draws a green box when a meteor is detected. It also buffers “before” and “after” frames so the saved clip includes the full event.

1) Internet + API key:
- `winMDL.py` is configured with a demo API key and model ID for quick testing. You can replace them with your own credentials later if you have a Roboflow account. These SHOULD work for the next couple of years (while I have a student account...)

2) Run it:
```bash
# From the project folder, with your venv activated
python winMDL.py
```
What you’ll see:
- A window titled “Webcam Meteor Detector” with boxes when detections happen. Press `q` to quit.
- Event videos are saved to `videos/` automatically.

## Where are my videos?
- Both scripts create a `videos/` folder in the project directory and save event clips as `.mp4` files.
- Filenames include timestamps, e.g., `20250101-235959.mp4` or `meteor_20250101_235959.mp4`.

## How detection works (no math)
- “Frames” are just single images from the webcam.
- The model looks for patterns in each frame that match “meteor-like” examples it learned earlier.
- A “confidence” is a score from 0 to 1—higher means the model is more sure. The local script uses a threshold (default 0.5). If the score is higher, it counts as a detection.
- Bounding boxes are green rectangles around things the model thinks are meteors.
- Pre- and post-buffers make sure the saved clip shows what happened just before and just after the detection, so you don’t miss context.

## Tuning behavior
- Local (`detector.py`):
  - Detection threshold: `DETECTION_THRESHOLD` (default `0.5`). Lower means more sensitive, but more false alarms.
  - Pre/post buffer sizes: `PRE_BUFFER_SIZE`, `POST_BUFFER_SIZE` (in frames). Increase to capture more context.
  - Output size/FPS: `FRAME_WIDTH/HEIGHT`, `FPS`.
- Cloud (`winMDL.py`):
  - Motion gating to save API calls: `MOTION_THRESHOLD_VALUE` and `MOTION_PIXEL_THRESHOLD` control when cloud inference runs.
  - Pre/post frames: `PRE_EVENT_FRAMES`, `POST_EVENT_FRAMES`.

## Troubleshooting
- “Error: Unable to open webcam.”
  - Close other apps using the camera.
  - Check camera permissions.
  - Try a different camera index in the code: `cv2.VideoCapture(1)` or `2`.

- “onnxruntime provider” error
  - Edit `detector.py` to use CPU only: `providers = ['CPUExecutionProvider']`.
  - Ensure `onnxruntime` is installed (no GPU package is needed on macOS): `pip install onnxruntime`.

- “ImportError: No module named cv2 / onnxruntime / inference_sdk”
  - Re-activate your virtual environment and reinstall: `pip install opencv-python numpy onnxruntime inference-sdk`.

- No videos appear
  - Make sure detections are actually happening. For testing, wave a flashlight or a bright object to simulate movement.
  - Increase sensitivity (lower the threshold in `detector.py`, or reduce `MOTION_PIXEL_THRESHOLD` in `winMDL.py`).

## Safety and usage notes
- Keep the camera pointed at the sky and avoid bright indoor lights to reduce false positives. Opt for wide-angle camera if possible.
- The cloud option sends frames over the internet—use only if you’re comfortable with that.
- Long runs can create many video files; check your disk space in `videos/`.

## Quick choices
- Want the simplest path? Try the cloud version first: `python winMDL.py`.
- Want offline detection or no internet usage? Use the local version with a `yolov5s.onnx` model: `python detector.py`.

Enjoy exploring the night sky! I hope you all continue to use and improve this over the years.
