# Dependencies and setup environment:
# Requires: ultralytics, torch, torchvision, opencv-python, numpy, pillow


# Import required libraries
import torch
import cv2
import csv
import numpy as np
from IPython.display import HTML
import os
import torchvision.transforms as T
from PIL import Image

# Load YOLOv10 model (same as before)
from ultralytics import YOLO
model = YOLO('best1.pt')  # Replace 'best.pt' with your trained YOLOv10 model path

# Load the depth estimation model (MiDaS model in this case)
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
depth_model.eval()
depth_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Depth transform to match the input of MiDaS
transform = T.Compose([T.Resize(384), T.ToTensor(), T.Normalize([0.5], [0.5])])

# Custom class mapping for instrument detection
class_mapping = {
    0: "ROI",
    1: "AtraumaticGrasper_Head",
    2: "AtraumaticGrasper_Body",
    3: "Electrichook",
    4: "CurvedAtraumaticGrasper_Head",
    5: "CurvedAtraumaticGrasper_Body",
    6: "Suction-Irrigation",
    7: "ClipApplier(Metal)_Head",
    8: "ClipApplier(Metal)_Body",
    9: "Scissors_Head",
    10: "Scissors_Body",
    11: "Overholt_Head",
    12: "Overholt_Body",
    13: "Ligasure_Head",
    14: "Ligasure_Body",
    15: "ClipApplier(Hem-O-Lok)_Head",
    16: "ClipApplier(Hem-O-Lok)_Body",
    17: "Specimanbag",
    # Add more classes as needed
}

# Particle filter class
class ParticleFilter:
    def __init__(self, num_particles, initial_position):
        self.num_particles = num_particles
        self.particles = np.random.randn(num_particles, 2) + initial_position
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, motion_model=np.array([0, 0]), noise=0.1):
        self.particles += motion_model + np.random.randn(self.num_particles, 2) * noise

    def update(self, detected_position):
        distances = np.linalg.norm(self.particles - detected_position, axis=1)
        self.weights = np.exp(-distances)

        if np.sum(self.weights) == 0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]

    def estimate(self):
        return np.mean(self.particles, axis=0)

# Function to calculate depth from the frame with added checks
def estimate_depth(frame, bbox):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        print(f"Invalid bounding box: {(x1, y1, x2, y2)}")
        return 0.0

    crop = frame[y1:y2, x1:x2]
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    crop_pil = crop_pil.resize((384, 384), Image.Resampling.LANCZOS)
    input_tensor = transform(crop_pil).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        depth_map = depth_model(input_tensor)

    depth_map = depth_map.squeeze().cpu().numpy()

    # Normalize depth values to the range of 0 to 1
    normalized_depth = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

    # Scale normalized depth to the expected range (e.g., 0 to 300 mm)
    scaled_depth = normalized_depth * 300  # Scale to 300 mm (30 cm)
    depth = np.mean(scaled_depth)

    print(f"Estimated normalized depth for bbox {bbox}: {depth:.2f} mm")
    return depth


# Detect objects using YOLOv10
def detect_objects(frame):
    results = model.predict(frame, imgsz=960, conf=0.6)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Coordinates of the box
            confidence = float(box.conf[0])  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            w, h = x2 - x1, y2 - y1
            detected_objects.append({
                "class_id": class_id,
                "class_name": class_mapping.get(class_id, "Unknown"),
                "box": (x1, y1, w, h),
                "confidence": confidence
            })

    return detected_objects

# Detect occlusion between instrument and ROI
def detect_occlusion(instrument_filter, roi_filter, threshold=3):
    instrument_position = instrument_filter.estimate()
    roi_position = roi_filter.estimate()
    distance = np.linalg.norm(instrument_position - roi_position)
    print(f"Distance between instrument and ROI: {distance}")
    return distance < threshold

# Initialize particle filters
instrument_filter = ParticleFilter(num_particles=100, initial_position=np.array([5, 5]))
roi_filter = ParticleFilter(num_particles=100, initial_position=np.array([10, 10]))

# Open the video file
cap = cv2.VideoCapture('video_1.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and output video writer
output_filename = 'output_video.mp4'
out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

csv_filename = "detection_results.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Class Name", "X", "Y", "Width", "Height", "Confidence", "Depth (mm)", "Occlusion"])

frame_count = 0

# Processing video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    detected_objects = detect_objects(frame)


    # Update particle filters based on detections
    for obj in detected_objects:
        class_id = obj["class_id"]
        class_name = class_mapping.get(class_id, "Unknown")
        x, y, w, h = obj["box"]
        confidence = obj["confidence"]
        depth = estimate_depth(frame, (x, y, x + w, y + h))

        if obj["class_id"] != 0:  # Class ID 1: Instrument
            detected_instrument_position = np.array([obj["box"][0] + obj["box"][2] / 2, obj["box"][1] + obj["box"][3] / 2])
            instrument_filter.update(detected_instrument_position)
        elif obj["class_id"] == 0:  # Class ID 0: ROI
            detected_roi_position = np.array([obj["box"][0] + obj["box"][2] / 2, obj["box"][1] + obj["box"][3] / 2])
            roi_filter.update(detected_roi_position)



    # Calculate and display occlusion based on depth for each instrument and ROI
    for obj in detected_objects:
        x, y, w, h = obj["box"]
        label = obj["class_name"]

        # Calculate depth for each instrument or ROI dynamically per frame
        # depth = estimate_depth(frame, (x, y, x + w, y + h))

        # Show depth as text next to the object
        text = f"{label} ({depth:.2f} mm)"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Check for occlusion
        if detect_occlusion(instrument_filter, roi_filter):
          cv2.putText(frame, "Warning: Occlusion Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
          occlusion_status = "Yes"
        else:
          occlusion_status="No"

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_count, class_name, x, y, w, h, confidence, depth, occlusion_status])

    # Predict next positions and resample particles
    instrument_filter.predict()
    roi_filter.predict()
    instrument_filter.resample()
    roi_filter.resample()

    # Write the frame to the output video
    out.write(frame)
    frame_count += 1

print(f"Total frames processed: {frame_count}")

# Release resources
cap.release()
out.release()

# Function to display the output video in Colab
#def show_video(file_name):
    #display(HTML(f"""
    #<video width="640" height="480" controls>
      #<source src="{file_name}" type="video/mp4">
    #</video>
    #"""))

# Show the processed video
#show_video(output_filename)
print(f"CSV file '{csv_filename}' successfully generated!")


