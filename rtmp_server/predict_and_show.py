import cv2
from ultralytics import YOLO
import torch
import time

# RTMP URL
rtmp_url = 'rtmp://0.0.0.0:1935/live'

# Check if MPS (Metal) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS (GPU) on macOS
    print("MPS is available and will be used for computation.")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("MPS is not available, using CPU.")

# Load the YOLOv8 model
model = YOLO('../models/yolov8n_custom_200_epoches_CPU_510_images.pt')
model.to(device)


# Check if the model is quantized
def check_model_quantization(model):
    quantized = True  # Assume it's quantized until proven otherwise
    for param in model.model.parameters():  # Access the model's parameters
        if param.dtype != torch.int8:  # Check if the parameter is not INT8
            quantized = False
            break
    return quantized


if check_model_quantization(model):
    print("The model is quantized.")
else:
    print("The model is not quantized.")

# Open the video stream
cap = cv2.VideoCapture(0)

# Check if the stream was opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

# Read and display frames from the stream
counter = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    counter += 1
    if counter % 4 != 0:
        continue

    # If the frame was read correctly, process it
    if ret:
        # Measure processing time for the frame
        frame_start_time = time.time()

        # Run YOLO model prediction
        results = model(frame)  # You can pass the frame directly to the model

        # Loop over the detections and draw bounding boxes
        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates and class predictions
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  # Convert to numpy array and then to integers
                x1, y1, x2, y2 = xyxy  # Bounding box (x1, y1, x2, y2)
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class index

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put label (class and confidence score)
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Measure the end time for frame processing
        frame_processing_time = time.time() - frame_start_time
        frame_count += 1

        # Calculate FPS every 10 frames
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")

        # Display the frame with bounding boxes
        cv2.imshow('RTMP Stream - YOLO Predictions', frame)
    else:
        print("Error: Could not read frame.")
        break

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
