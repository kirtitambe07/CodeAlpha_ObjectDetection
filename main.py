import cv2
import time
from ultralytics import YOLO

# Configuration
SOURCE = 0 # 0 for default webcam, or provide a string path to a video file e.g. "video.mp4"
MODEL_NAME = "yolov8n.pt"
OUTPUT_FILE = "output.mp4"

def main():
    # Load the YOLOv8 model
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Open the video source
    cap = cv2.VideoCapture(SOURCE, cv2.CAP_DSHOW) if isinstance(SOURCE, int) else cv2.VideoCapture(SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {SOURCE}")
        return

    # Get video properties for the output video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:  # sometimes webcam returns 0 FPS
        fps = 30
        
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (frame_width, frame_height))

    # Initialize variables for FPS and object counting
    prev_time = 0
    unique_ids = set()

    print("Starting video stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of stream or error reading frame.")
            break

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # Get the annotated frame
        annotated_frame = results[0].plot()

        # Update unique IDs
        if results[0].boxes.id is not None:
            boxes_ids = results[0].boxes.id.int().cpu().tolist()
            unique_ids.update(boxes_ids)

        # Calculate FPS
        current_time = time.time()
        fps_text = f"FPS: {1 / (current_time - prev_time + 1e-9):.1f}"
        prev_time = current_time

        # Draw FPS and unique object count
        cv2.putText(annotated_frame, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Unique Objects: {len(unique_ids)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(annotated_frame)

        # Display the frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
