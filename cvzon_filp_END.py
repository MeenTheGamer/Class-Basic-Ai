from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np

# Initialize the video capture with lower resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate to 30 FPS

# Load the YOLO model
model = YOLO(r'D:\ClassAi\AI_Med_RSU-main\yolov8n.pt')

# List of class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

while True:
    # Capture frame-by-frame
    success, img = cap.read()

    # Run the YOLO model on the image
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            # Draw the bounding box with corners
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=3, colorR=(255, 0, 255))

            # Get confidence level and class name
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = box.cls[0]
            name = classNames[int(cls)]

            # Create the text
            text = f'{name} {conf}'

            # Calculate the width and height of the text box
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = max(0, x1)
            text_y = max(35, y1)
            
            # Create a blank image for the text box
            text_img = np.zeros((text_height + baseline, text_width, 3), dtype=np.uint8)
            
            # Put the text on the blank image
            cv2.putText(text_img, text, (0, text_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Flip the text image horizontally
            text_img = cv2.flip(text_img, 1)

            # Ensure text_img fits within the bounds of img
            end_x = min(text_x + text_width, img.shape[1])
            end_y = min(text_y, img.shape[0])
            
            # Get the region of interest on the original image
            roi = img[text_y - text_height - baseline:end_y, text_x:end_x]

            # Adjust text_img to fit the roi size
            text_img = text_img[:roi.shape[0], :roi.shape[1]]

            # Add the text image to the ROI
            result = cv2.addWeighted(roi, 1, text_img, 1, 0)

            # Place the result back on the original image
            img[text_y - text_height - baseline:end_y, text_x:end_x] = result

    # Flip the image horizontally
    img_hand = cv2.flip(img, 1)

    # Display the resulting frame
    cv2.imshow("Image", img_hand)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
