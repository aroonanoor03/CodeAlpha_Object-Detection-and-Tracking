import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

class DeepSORTTracker:
    """
    A simplified implementation of Deep SORT tracking algorithm
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_history = defaultdict(lambda: [])
    
    def update(self, detections):
        """
        Update tracker with new detections
        """
        # Predict new locations for existing tracks
        for track in self.tracks:
            track['age'] += 1
        
        # Match detections to existing tracks using IoU
        matches, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx]['bbox'] = detections[det_idx]['bbox']
            self.tracks[track_idx]['age'] = 0
            self.tracks[track_idx]['class_id'] = detections[det_idx]['class_id']
            self.tracks[track_idx]['confidence'] = detections[det_idx]['confidence']
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = {
                'id': self.next_id,
                'bbox': detections[det_idx]['bbox'],
                'age': 0,
                'class_id': detections[det_idx]['class_id'],
                'confidence': detections[det_idx]['confidence']
            }
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove dead tracks
        self.tracks = [track for track in self.tracks if track['age'] < self.max_age]
        
        # Update track history
        for track in self.tracks:
            if track['age'] == 0:  # Only update if track was matched
                center = self._get_center(track['bbox'])
                self.track_history[track['id']].append(center)
                if len(self.track_history[track['id']]) > 50:  # Keep last 50 points
                    self.track_history[track['id']].pop(0)
        
        return self.tracks
    
    def _match_detections_to_tracks(self, detections):
        """
        Match detections to tracks using IoU metric
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track['bbox'], det['bbox'])
        
        # Find matches where IoU > threshold
        matches = []
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                if iou_matrix[i, j] > self.iou_threshold:
                    matches.append((i, j))
        
        # Find unmatched detections and tracks
        matched_detections = set(j for _, j in matches)
        matched_tracks = set(i for i, _ in matches)
        
        unmatched_detections = [j for j in range(len(detections)) if j not in matched_detections]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate coordinates of intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        return iou
    
    def _get_center(self, bbox):
        """
        Get center point of bounding box
        """
        x, y, w, h = bbox
        return (int(x + w/2), int(y + h/2))

class ObjectDetectionTracker:
    """
    Main class for object detection and tracking
    """
    def __init__(self, video_path, model_path='yolov8n.pt', confidence_threshold=0.5):
        self.video_path = video_path
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)
        self.tracker = DeepSORTTracker()
        self.cap = None
        self.class_names = self.model.names
    
    def initialize_video(self):
        """
        Initialize video capture
        """
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video loaded: {self.width}x{self.height}, FPS: {self.fps}, Total frames: {self.total_frames}")
    
    def process_frame(self, frame):
        """
        Process a single frame: detect objects and track them
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = box.conf[0].item()
                    if confidence > self.confidence_threshold:
                        class_id = int(box.cls[0].item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        w, h = x2 - x1, y2 - y1
                        
                        detections.append({
                            'bbox': (x1, y1, w, h),
                            'confidence': confidence,
                            'class_id': class_id
                        })
        
        # Update tracker with detections
        tracks = self.tracker.update(detections)
        
        # Draw results on frame
        annotated_frame = self.draw_results(frame, tracks)
        
        return annotated_frame, tracks
    
    def draw_results(self, frame, tracks):
        """
        Draw bounding boxes, labels, confidence scores, and tracking IDs on frame
        """
        annotated_frame = frame.copy()
        
        for track in tracks:
            x, y, w, h = track['bbox']
            class_id = track['class_id']
            confidence = track['confidence']
            track_id = track['id']
            
            # Draw bounding box
            color = self.get_color(track_id)
            cv2.rectangle(annotated_frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            
            # Draw label background
            label = f"{self.class_names[class_id]} {confidence:.2f} ID:{track_id}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(annotated_frame, (int(x), int(y) - label_height - 5), 
                         (int(x) + label_width, int(y)), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (int(x), int(y) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw track history
            if track_id in self.tracker.track_history:
                track_points = self.tracker.track_history[track_id]
                for i in range(1, len(track_points)):
                    cv2.line(annotated_frame, track_points[i-1], track_points[i], color, 2)
        
        return annotated_frame
    
    def get_color(self, track_id):
        """
        Generate a consistent color for each track ID
        """
        np.random.seed(track_id)
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        return color
    
    def run(self):
        """
        Main processing loop
        """
        self.initialize_video()
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, tracks = self.process_frame(frame)
            
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Objects: {len(tracks)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Object Detection and Tracking', processed_frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"Processing complete. Processed {frame_count} frames in {elapsed_time:.2f} seconds")

def main():
    """
    Main function to run the object detection and tracking pipeline
    """
    # Configuration-
    VIDEO_PATH = r"C:853889-hd_1920_1080_25fps.mp4"
    MODEL_PATH = "yolov8n.pt" 
    CONFIDENCE_THRESHOLD = 0.5
    
    try:
        # Create and run tracker
        tracker = ObjectDetectionTracker(VIDEO_PATH, MODEL_PATH, CONFIDENCE_THRESHOLD)
        tracker.run()
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. A valid video file at the specified path")
        print("2. Internet connection for model download (if first run)")
        print("3. All required packages installed")

if __name__ == "__main__":
    main()