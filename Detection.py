import cv2
import numpy as np
import sqlite3
from datetime import datetime

# Initialize the SQLite database
conn = sqlite3.connect('detection_database.db')
cursor = conn.cursor()

# Create a table to store detected objects
cursor.execute('''CREATE TABLE IF NOT EXISTS object_detections
                (timestamp DATETIME, object_label TEXT, confidence REAL)''')

# Create a table to store detected faces
cursor.execute('''CREATE TABLE IF NOT EXISTS face_detections
                (timestamp DATETIME, x INT, y INT, width INT, height INT)''')

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names for object detection
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load pre-trained face recognition models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to detect humans and objects
def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)

    return outs, classes

# Function to draw bounding boxes
def draw_boxes(frame, outs, classes):
    height, width, _ = frame.shape

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                label = str(classes[class_id])
                color = (0, 255, 0) if label != 'person' else (255, 0, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Store object detection in the database
                timestamp = datetime.now()
                cursor.execute("INSERT INTO object_detections (timestamp, object_label, confidence) VALUES (?, ?, ?)",
                               (timestamp, label, confidence))
                conn.commit()

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for system camera, replace with the appropriate source for CCTV or other cameras

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    outs, classes = detect_objects(frame)

    # Draw bounding boxes around detected objects
    draw_boxes(frame, outs, classes)

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Store face detection in the database
        timestamp = datetime.now()
        cursor.execute("INSERT INTO face_detections (timestamp, x, y, width, height) VALUES (?, ?, ?, ?, ?)",
                       (timestamp, x, y, w, h))
        conn.commit()

    # Display the frame
    cv2.imshow('Video Feed', frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera, close the database connection, and close all windows
cap.release()
conn.close()
cv2.destroyAllWindows()
