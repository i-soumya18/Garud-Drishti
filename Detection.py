import cv2
import numpy as np
import sqlite3

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize variables for object tracking
object_ids = {}
counter = 0

# Create a connection to the database
conn = sqlite3.connect("crowd_data.db")
cursor = conn.cursor()

# Create a table to store data
cursor.execute('''
    CREATE TABLE IF NOT EXISTS detected_humans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        height REAL,
        human_id INTEGER,
        dress_color TEXT
    )
''')
# Load pre-trained face recognition models
known_face_encodings = []
known_face_names = []

# Function to detect and recognize humans
def detect_and_recognize_faces(frame):
    # Convert the frame to RGB format (required by face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match is found, use the name associated with the matched face
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Draw a rectangle around the detected face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Insert the detected face into the database (assuming it's a new face)
        if name == "Unknown":
            cursor.execute("INSERT INTO detected_humans (name, encoding) VALUES (?, ?)", (name, face_encoding.tobytes()))
            conn.commit()

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

    return frame

# Function to insert data into the database
def insert_data(height, human_id, dress_color):
    cursor.execute("INSERT INTO detected_humans (height, human_id, dress_color) VALUES (?, ?, ?)", (height, human_id, dress_color))
    conn.commit()

# Function to detect crowd
def detect_crowd(frame, num_threshold):
    outs, classes = detect_objects(frame)
    num_people = sum(1 for cls in classes if cls == 'person')

    return num_people >= num_threshold

# Function to detect motion between frames
def detect_movement(previous_frame, current_frame, threshold=10000):
    outs_prev, _ = detect_objects(previous_frame)
    outs_current, _ = detect_objects(current_frame)

    # Calculate the Euclidean distance between the detected human centroids
    centroid_prev = None
    centroid_current = None

    for out in outs_prev:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'person':
                x, y, w, h = (detection[:4] * np.array(
                    [current_frame.shape[1], current_frame.shape[0], current_frame.shape[1],
                     current_frame.shape[0]])).astype(int)
                centroid_prev = (x + w // 2, y + h // 2)

    for out in outs_current:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'person':
                x, y, w, h = (detection[:4] * np.array(
                    [current_frame.shape[1], current_frame.shape[0], current_frame.shape[1],
                     current_frame.shape[0]])).astype(int)
                centroid_current = (x + w // 2, y + h // 2)

    # Check if the distance between centroids exceeds the threshold
    if centroid_prev is not None and centroid_current is not None:
        distance = np.linalg.norm(np.array(centroid_prev) - np.array(centroid_current))
        return distance > threshold
    else:
        return False

# Function to assign unique IDs to detected humans
def assign_unique_ids(frame):
    global counter
    outs, _ = detect_objects(frame)

    # Initialize a dictionary to store current frame's object IDs
    current_frame_ids = {}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'person':
                x, y, w, h = (detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)

                # Calculate the centroid of the detected person
                centroid = (x + w // 2, y + h // 2)

                # Check if the centroid is close to any existing object
                object_id = None
                for obj_id, obj_centroid in object_ids.items():
                    distance = np.linalg.norm(np.array(centroid) - np.array(obj_centroid))
                    if distance < 50:  # Adjust this threshold as needed
                        object_id = obj_id
                        break

                # Assign a new object ID if not close to any existing object
                if object_id is None:
                    object_id = counter
                    counter += 1

                # Update the dictionary with the current object's ID and centroid
                current_frame_ids[object_id] = centroid

                # Draw the object's ID on the frame
                cv2.putText(frame, str(object_id), (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, centroid, 4, (0, 255, 0), -1)

    # Update the global object_ids dictionary with the current frame's IDs
    object_ids.clear()
    object_ids.update(current_frame_ids)

    return frame

# Example usage with real-time video input
cap = cv2.VideoCapture(0)  # Use 0 for system camera, replace with the appropriate source for CCTV or satellite
previous_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_ids = assign_unique_ids(frame)
    cv2.imshow('Object Tracking', frame_with_ids)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()