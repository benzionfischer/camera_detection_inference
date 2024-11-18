import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# RTMP URL
rtmp_url = 'rtmp://0.0.0.0:1935/live'

# Check if MPS (Metal) is available
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS (GPU) on macOS
    print("MPS is available and will be used for computation.")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("MPS is not available, using CPU.")

# Load the TensorFlow Lite model
model_path = '../models/yolov8n_custom_200_epoches_CPU_510_images_float16.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
        # Resize the frame to the required input size for YOLOv8 model (e.g., 640x640)
        input_shape = input_details[0]['shape']
        input_size = (input_shape[1], input_shape[2])
        resized_frame = cv2.resize(frame, input_size)

        # Normalize and prepare input data
        input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32)
        input_data = np.array(input_data)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Process the results
        boxes = output_data[0][:, :4]  # Get bounding boxes (x1, y1, x2, y2)
        confidences = output_data[0][:, 4]  # Get confidence scores
        class_ids = output_data[0][:, 5].astype(int)  # Get class IDs

        # Loop over the detections and draw bounding boxes
        for i in range(len(boxes)):
            if confidences[i] > 0.5:  # Filter out low-confidence detections
                x1, y1, x2, y2 = boxes[i]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = confidences[i]
                cls = class_ids[i]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put label (class and confidence score)
                label = f'Class {cls} {conf:.2f}'
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
