import cv2
from ultralytics import YOLO
import torch
import time


# roubleshoot incoming traffic to mac
# sudo tcpdump -i any -n port 1935


# RTMP URL
rtmp_url = 'rtmp://0.0.0.0:1935/live'

# Load the YOLOv8 model (make sure you have the correct path to your fine-tuned model)

# Check if MPS (Metal) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS (GPU) on macOS
    print("MPS is available and will be used for computation.")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("MPS is not available, using CPU.")

model = YOLO('../yolov8n_custom_200_epoches_CPU_510_images.pt')
model.to(device)


# Open the video stream
cap = cv2.VideoCapture(rtmp_url)

# Check if the stream was opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

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
