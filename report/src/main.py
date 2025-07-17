import cv2
import argparse
from ultralytics import YOLO
from tracker import PlayerTracker

def main(video_path, model_path, output_path):
    """
    Main function to process the video, perform player tracking, and save the result.
    """
    # 1. Initialize Models
    detection_model = YOLO(model_path)
    
    # 2. Initialize Tracker
    # The tracker will handle state management (Kalman Filter) and Re-ID (appearance features)
    tracker = PlayerTracker(
        reid_model_path='osnet_x0_25_msmt17.pt', # Example of a lightweight ReID model
        max_age=30, # Max frames to keep a track without detection
        reid_threshold=0.85 # Cosine similarity threshold for re-identification
    )

    # 3. Video I/O
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Get Detections
        # We are interested in class '0' which we assume are players
        results = detection_model(frame, classes=[0], verbose=False)
        detections = results[0].boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]

        # 5. Update Tracker
        # The tracker's update method contains the core logic:
        # - Predict next state for existing tracks
        # - Match detections to tracks (using motion and appearance)
        # - Update matched tracks
        # - Re-identify lost tracks
        # - Initialize new tracks
        tracked_players = tracker.update(detections, frame)

        # 6. Visualize Results
        # Draw bounding boxes and IDs on the frame
        for player in tracked_players:
            x1, y1, x2, y2, player_id = player
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {int(player_id)}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)
        print(f"Processed frame {frame_num}")
        frame_num += 1

    # 7. Release Resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Player Re-Identification in Sports Footage")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO detection model.")
    parser.add_argument("--output_path", type=str, default="output/result.mp4", help="Path to save the output video.")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    main(args.video_path, args.model_path, args.output_path)
