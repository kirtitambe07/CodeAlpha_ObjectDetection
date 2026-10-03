# CodeAlpha AI Internship - Task 4: Object Detection and Tracking

## Description
This project demonstrates real-time object detection and tracking using a webcam or video file. It leverages the YOLOv8 model for fast and accurate object detection, and ByteTrack for assigning and maintaining unique IDs for each detected object across frames.

## Features
- Real-time object detection using YOLOv8n (Nano).
- Object tracking using the robust ByteTrack algorithm.
- Displays FPS (Frames Per Second) and the total number of unique objects tracked.
- Saves the annotated tracking session to `output.mp4`.
- Graceful error handling for missing video sources.

## How It Works
1. **Detection**: YOLOv8 scans each frame of the video and identifies objects (like people, cars, etc.) along with their bounding boxes and classes.
2. **Tracking**: The SORT-family tracker (ByteTrack) analyzes the bounding boxes from consecutive frames. It matches objects based on their positions and movements to assign a persistent ID, allowing us to track them even if they briefly get occluded.
3. **Annotation**: The script draws the bounding boxes, classes, and IDs onto the frame, computes the FPS, and displays the running total of unique objects.

## Installation
1. Clone the repository or create the project folder.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
Execute the main script:
```bash
python main.py
```
Press `q` at any time while the window is focused to exit the application gracefully.

## Demo
*(Add a screenshot or GIF of the output here)*
