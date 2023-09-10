import cv2
import numpy as np
import sqlite3
from datetime import datetime

# Initialize the SQLite database
try:
    conn = sqlite3.connect('detection_database.db')
    cursor = conn.cursor()
except sqlite3.Error as e:
    print("SQLite Error:", e)
    exit(1)

# Create tables (if not exists) for detected objects and faces
try:
    cursor.execute('''CREATE TABLE IF NOT EXISTS object_detections
                    (timestamp DATETIME, object_label TEXT, confidence REAL)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS face_detections
                    (timestamp DATETIME, x INT, y INT, width INT, height INT, unique_id INT)''')
except sqlite3.Error as e:
    print("SQLite Error:", e)
    conn.close()
    exit(1)

# Load YOLOv3 model
try:
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
except cv2.error as e:
    print("OpenCV Error:", e)
    conn.close()
    exit(1)

# Load COCO class names for object detection
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load pre-trained face recognition models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize variables for tracking unique IDs
next_unique_id = 1
detected_faces = {}  # Dictionary to store detected faces and their IDs
import time

# Initialize variables for frame rate calculation
prev_time = time.time()
frame_count = 0
frame_rate = 0

def calculate_frame_rate():
    global prev_time, frame_count, frame_rate
    current_time = time.time()
    frame_count += 1
    elapsed_time = current_time - prev_time

    if elapsed_time >= 1.0:  # Update frame rate every 1 second
        frame_rate = frame_count / elapsed_time
        frame_count = 0
        prev_time = current_time

    return frame_rate
# Function to calculate dress color and height of a person based on detected region
# Initialize variables for tracking unique IDs
next_unique_id = 1
detected_persons = {}  # Dictionary to store detected persons and their IDs

# Function to calculate dress color and height of a person based on detected region
def calculate_dress_color_and_height(frame, x, y, w, h):
    # Crop the region around the person's chest for dress color detection
    chest_region = frame[y:y + int(1.2 * h), x:x + w]

    # Calculate the average color of the chest region (you can use a more sophisticated method for dress color)
    avg_color = np.mean(chest_region, axis=(0, 1))
    dress_color = "Unknown"  # Placeholder for dress color detection

    # Calculate height based on the person's bounding box
    height = h  # You can refine this based on your camera perspective and calibration

    return dress_color, height

# Function to detect humans and objects
def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)
    return outs, classes

# Function to draw bounding boxes and assign unique IDs to humans
def draw_boxes(frame, outs, classes):
    global next_unique_id, detected_persons  # Declare global variables

    height, width, _ = frame.shape
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                label = str(classes[class_id])
                if label == 'person':
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Store object detection in the database
                    timestamp = datetime.now()
                    try:
                        cursor.execute("INSERT INTO object_detections (timestamp, object_label, confidence) VALUES (?, ?, ?)",
                                       (timestamp, label, confidence))
                        conn.commit()
                    except sqlite3.Error as e:
                        print("SQLite Error:", e)

                    # Check if the detected person is already in the dictionary
                    found_match = False
                    for person_id, (prev_x, prev_y, prev_w, prev_h) in detected_persons.items():
                        if x > prev_x and y > prev_y and x + w < prev_x + prev_w and y + h < prev_y + prev_h:
                            # Detected person matches a previously detected person
                            found_match = True
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, f'Person {person_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            break

                    if not found_match:
                        # This is a new person, assign a unique ID
                        detected_persons[next_unique_id] = (x, y, w, h)
                        dress_color, height = calculate_dress_color_and_height(frame, x, y, w, h)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f'Person {next_unique_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        timestamp = datetime.now()
                        try:
                            cursor.execute("INSERT INTO person_detections (timestamp, unique_id, dress_color, height) VALUES (?, ?, ?, ?)",
                                           (timestamp, next_unique_id, dress_color, height))
                            conn.commit()
                        except sqlite3.Error as e:
                            print("SQLite Error:", e)
                        next_unique_id += 1

cap = cv2.VideoCapture(0)  # Use 0 for system camera, replace with the appropriate source for CCTV or other cameras

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    outs, classes = detect_objects(frame)

    # Draw bounding boxes around detected objects and assign unique IDs to humans
    draw_boxes(frame, outs, classes)

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces and assign unique IDs to faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Check if the detected face matches a previously detected person
        found_match = False
        for face_id, (prev_x, prev_y, prev_w, prev_h) in detected_faces.items():
            if x > prev_x and y > prev_y and x + w < prev_x + prev_w and y + h < prev_y + prev_h:
                # Detected face matches a previously detected person
                found_match = True
                cv2.putText(frame, f'Person {face_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                break

        if not found_match:
            # This is a new face, assign a unique ID
            detected_faces[next_unique_id] = (x, y, w, h)
            cv2.putText(frame, f'Person {next_unique_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            timestamp = datetime.now()
            try:
                cursor.execute("INSERT INTO face_detections (timestamp, x, y, width, height, unique_id) VALUES (?, ?, ?, ?, ?, ?)",
                               (timestamp, x, y, w, h, next_unique_id))
                conn.commit()
            except sqlite3.Error as e:
                print("SQLite Error:", e)
            next_unique_id += 1

    # Calculate and display frame rate
    frame_rate = calculate_frame_rate()
    cv2.putText(frame, f'Frame Rate: {frame_rate:.2f} FPS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Video Feed', frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the camera, close the database connection, and close all windows
cap.release()
conn.close()
cv2.destroyAllWindows()
