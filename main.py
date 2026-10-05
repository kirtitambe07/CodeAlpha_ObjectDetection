import cv2
import time
import csv
from collections import defaultdict, deque
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
SOURCE = 0                          # 0 for webcam, or string path to video
MODEL_PATH = "yolov8n.pt"           # YOLOv8 model path
OUTPUT_FILE = "output.mp4"          # Output video file
CONF_THRESHOLD = 0.6                # Minimum confidence for detection
MIN_FRAMES = 45                     # Frames needed to confirm an object
TRAIL_LENGTH = 30                   # How many previous positions to remember
# Define the restricted zone as fractions of the screen width and height
# Format: (x_min, y_min, x_max, y_max)
ZONE_FRACTION = (0.60, 0.20, 0.98, 0.95)
ALERT_COOLDOWN_SECONDS = 2

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_color(track_id):
    """Generate a consistent color based on the track ID."""
    np.random.seed(int(track_id))
    # Return a BGR tuple
    return tuple([int(c) for c in np.random.randint(0, 255, size=3)])

def is_inside_zone(point, zone_rect):
    """Check if a point (x, y) is inside the given rectangle."""
    x, y = point
    x1, y1, x2, y2 = zone_rect
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_text_with_outline(img, text, position, font_scale, color, thickness):
    """Draw text with a black outline for better readability."""
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def save_logs(tracks_info, class_names):
    """Save the detection logs to a CSV and generate a summary chart."""
    csv_file = "detections_log.csv"
    confirmed_tracks = [t for t in tracks_info.values() if t['frames'] >= MIN_FRAMES]
    
    # 1. Save CSV
    try:
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "class", "first_seen_s", "seconds_in_view", "frames"])
            for t in confirmed_tracks:
                writer.writerow([
                    t['id'], 
                    class_names[t['class_id']], 
                    f"{t['first_seen_s']:.2f}", 
                    f"{t['seconds_in_view']:.2f}", 
                    t['frames']
                ])
        print(f"Saved detection logs to {csv_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # 2. Save Matplotlib Chart
    try:
        class_counts = defaultdict(int)
        for t in confirmed_tracks:
            class_names_str = class_names[t['class_id']]
            class_counts[class_names_str] += 1
            
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            plt.figure(figsize=(8, 6))
            plt.bar(classes, counts, color='skyblue')
            plt.xlabel('Object Class')
            plt.ylabel('Confirmed Count')
            plt.title('Confirmed Objects per Class')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("summary.png")
            plt.close()
            print("Saved summary chart to summary.png")
    except Exception as e:
        print(f"Error saving chart: {e}")


def main():
    # Load YOLO model
    try:
        model = YOLO(MODEL_PATH)
        class_names = model.names
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Open the video source
    cap = cv2.VideoCapture(SOURCE, cv2.CAP_DSHOW) if isinstance(SOURCE, int) else cv2.VideoCapture(SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {SOURCE}")
        return

    # Get video properties for saving
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: 
        fps = 30
        
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Convert zone fractions to pixel coordinates
    zx1 = int(ZONE_FRACTION[0] * frame_width)
    zy1 = int(ZONE_FRACTION[1] * frame_height)
    zx2 = int(ZONE_FRACTION[2] * frame_width)
    zy2 = int(ZONE_FRACTION[3] * frame_height)
    zone_rect = (zx1, zy1, zx2, zy2)

    # State Variables
    tracks_info = {}       # Dictionary to store info for each track_id
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    zone_alerts_count = 0
    start_time = time.time()
    last_person_in_zone_time = 0.0
    
    # For smoothed FPS
    fps_history = deque(maxlen=10)
    prev_time = time.time()

    print("Starting Smart Monitoring System. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = time.time()
            elapsed_sys_time = current_time - start_time
            
            # FPS Calculation
            fps_history.append(1.0 / (current_time - prev_time + 1e-9))
            prev_time = current_time
            smoothed_fps = sum(fps_history) / len(fps_history)
            
            # Run YOLOv8 Tracking
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD, verbose=False)
            
            # Create a copy for annotation
            annotated_frame = frame.copy()
            
            # Live counts
            live_class_counts = defaultdict(int)
            alert_active = False

            # Draw the semi-transparent zone (default green)
            zone_color = (0, 255, 0)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Update object metadata (dwell time, frames)
                    if track_id not in tracks_info:
                        tracks_info[track_id] = {
                            'id': track_id,
                            'class_id': class_id,
                            'first_seen_s': elapsed_sys_time,
                            'seconds_in_view': 0.0,
                            'frames': 0,
                            'in_zone': False
                        }
                    
                    # Update time and frame count
                    tracks_info[track_id]['seconds_in_view'] = elapsed_sys_time - tracks_info[track_id]['first_seen_s']
                    tracks_info[track_id]['frames'] += 1
                    
                    # Calculate center point for zone checking and trail
                    center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    
                    # Add to trails
                    trails[track_id].append(center_point)
                    
                    live_class_counts[class_names[class_id]] += 1
                    
                    # Check Zone Intrusion
                    is_person = (class_names[class_id] == "person")
                    in_zone = is_inside_zone(center_point, zone_rect)
                    
                    box_color = get_color(track_id)
                    
                    if is_person and in_zone:
                        alert_active = True
                        zone_color = (0, 0, 255) # Red zone
                        box_color = (0, 0, 255)  # Red box
                        
                        # Count entry
                        if not tracks_info[track_id]['in_zone']:
                            tracks_info[track_id]['in_zone'] = True
                    else:
                        tracks_info[track_id]['in_zone'] = False
                        
                    # Draw Trail
                    track_trail = trails[track_id]
                    for i in range(1, len(track_trail)):
                        cv2.line(annotated_frame, track_trail[i-1], track_trail[i], get_color(track_id), 2)
                        
                    # Draw Box and Label
                    label = f"{class_names[class_id]} id:{track_id} {int(tracks_info[track_id]['seconds_in_view'])}s"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                    draw_text_with_outline(annotated_frame, label, (x1, label_y), 0.5, box_color, 2)
            
            if alert_active:
                if current_time - last_person_in_zone_time >= ALERT_COOLDOWN_SECONDS:
                    zone_alerts_count += 1
                last_person_in_zone_time = current_time
            
            # Draw the restricted zone
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), zone_color, -1)
            # Apply transparency (0.3 alpha)
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            cv2.rectangle(annotated_frame, (zx1, zy1), (zx2, zy2), zone_color, 2)
            
            # Draw UI Elements
            # Confirmed Objects Count
            confirmed_count = sum(1 for t in tracks_info.values() if t['frames'] >= MIN_FRAMES)
            
            # Top-left display texts
            text_color = (0, 255, 0)  # Bright green
            
            # Draw semi-transparent background for info panel
            num_lines = 3 + len(live_class_counts)
            panel_height = num_lines * 30 + 10
            panel_width = 320
            panel_overlay = annotated_frame.copy()
            cv2.rectangle(panel_overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(panel_overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            
            y_offset = 30
            draw_text_with_outline(annotated_frame, f"FPS: {smoothed_fps:.1f}", (10, y_offset), 0.7, text_color, 2)
            y_offset += 30
            draw_text_with_outline(annotated_frame, f"Confirmed objects: {confirmed_count}", (10, y_offset), 0.7, text_color, 2)
            y_offset += 30
            draw_text_with_outline(annotated_frame, f"Zone alerts: {zone_alerts_count}", (10, y_offset), 0.7, text_color, 2)
            
            for cls_name, count in live_class_counts.items():
                y_offset += 30
                draw_text_with_outline(annotated_frame, f"{cls_name}: {count}", (10, y_offset), 0.6, text_color, 2)
                
            # Alert message
            if alert_active:
                draw_text_with_outline(annotated_frame, "ALERT: PERSON IN RESTRICTED ZONE", (10, frame_height - 30), 1.0, (0, 0, 255), 3)
                
            out.write(annotated_frame)
            cv2.imshow("Smart Monitoring System", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Run upon normal exit or crash
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Output saved to {OUTPUT_FILE}")
        save_logs(tracks_info, model.names)

if __name__ == "__main__":
    main()
