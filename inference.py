import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

# Check if MPS (Metal) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available and will be used for computation.")
else:
    device = torch.device("cpu")
    print("MPS is not available, using CPU.")

# Load the YOLO model
model_path = 'yolov8n_custom_200_epoches_CPU_510_images_ncnn_model'
tflite_model = YOLO(model_path)

# OpenCV to capture video
cap = cv2.VideoCapture(0)  # Use webcam input

counter = 0
while True:

    # Start timer for inference
    start_time = time.time()

    ret, frame = cap.read()

    # Stop timer and calculate duration
    opencv_time = time.time() - start_time

    if not ret:
        break


    # counter += 1
    # if counter % 6 != 0:
    #     continue

    # Start timer for inference
    start_time = time.time()

    # Perform inference
    results = tflite_model(frame)

    # Stop timer and calculate duration
    inference_time = time.time() - start_time
    print(f"opencv Time: {opencv_time:.4f} seconds")
    print(f"Inference Time: {inference_time:.4f} seconds")

    # Start timer for inference
    start_time = time.time()

    # Process each detection
    for result in results[0].boxes:  # Access boxes from results
        # Convert the result to numpy array
        box = result.xyxy[0].cpu().numpy()

        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box[:4])

        # Extract confidence and class_id if available
        conf = float(result.conf[0]) if hasattr(result, 'conf') else 1.0
        class_id = int(result.cls[0]) if hasattr(result, 'cls') else -1

        # Get label if class_id exists
        label = results[0].names[class_id] if class_id >= 0 else "Unknown"

        # Draw the bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Inference', frame)

    # Stop timer and calculate duration
    presentation_time = time.time() - start_time
    print(f"Presentation Time: {presentation_time:.4f} seconds")

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
