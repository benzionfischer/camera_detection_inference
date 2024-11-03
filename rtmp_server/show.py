import cv2

# RTMP URL
rtmp_url = 'rtmp://0.0.0.0:1935/live'

# Open the video stream
cap = cv2.VideoCapture(rtmp_url)

# Check if the stream was opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Read and display frames from the stream
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was read correctly, display it
    if ret:
        cv2.imshow('RTMP Stream', frame)
    else:
        print("Error: Could not read frame.")
        break

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
