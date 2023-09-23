import cv2
import numpy as np
import sqlite3
from datetime import datetime
import winsound  # Import the winsound library for playing alert sound
from flask import jsonify
from mtcnn import MTCNN

from email_alert import send_email

# Initialize the MTCNN detector
mtcnn_detector = MTCNN()


#Function to add current time and date to the camera feed frame
def add_timestamp(frame):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS
    cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Display at (10, 30) with white color

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

# Initialize the SQLite database for storing human information
try:
    conn_human_info = sqlite3.connect('human_info_database.db')
    cursor_human_info = conn_human_info.cursor()
except sqlite3.Error as e:
    print("SQLite Error:", e)
    exit(1)

# Create a table for storing human information (e.g., name, age, etc.)
try:
    cursor_human_info.execute('''CREATE TABLE IF NOT EXISTS human_info
                                (unique_id INT, name TEXT, age INT, gender TEXT)''')
except sqlite3.Error as e:
    print("SQLite Error:", e)
    conn_human_info.close()
    exit(1)

# Function to check if a detected face matches any entry in the human information database
def check_detected_face_in_database(unique_id):
    try:
        cursor_human_info.execute("SELECT * FROM human_info WHERE unique_id=?", (unique_id,))
        result = cursor_human_info.fetchone()
        if result:
            return result  # Return the information about the detected human
        else:
            return None  # No match found in the database
    except sqlite3.Error as e:
        print("SQLite Error:", e)
        return None

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

# Create the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

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
# Function to play an alert sound
def play_alert_sound():
    # Play a beep sound for 1 second (you can replace this with your preferred alert sound)
    winsound.Beep(1000, 1000)


def stop_system():
    try:
        if cv2.waitKey(10) & 0xFF == ord('q'):

            # For example, you can release the camera and perform cleanup operations
            cap.release()  # Release the camera
            conn.close()   # Close the SQLite database connection

        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e))



# Function to draw bounding boxes

def draw_boxes(frame, outs, classes):
    global next_unique_id  # Declare next_unique_id as a global variable

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
                    for face_id, (prev_x, prev_y, prev_w, prev_h) in detected_faces.items():
                        if x > prev_x and y > prev_y and x + w < prev_x + prev_w and y + h < prev_y + prev_h:
                            # Detected person matches a previously detected face
                            found_match = True
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, f'Person {face_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            break

                    if not found_match:
                        # This is a new person, assign a unique ID
                        detected_faces[next_unique_id] = (x, y, w, h)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f'Person {next_unique_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        timestamp = datetime.now()
                        try:
                            cursor.execute("INSERT INTO face_detections (timestamp, x, y, width, height, unique_id) VALUES (?, ?, ?, ?, ?, ?)",
                                           (timestamp, x, y, w, h, next_unique_id))
                            conn.commit()
                        except sqlite3.Error as e:
                            print("SQLite Error:", e)
                        next_unique_id += 1

                    # Detect and draw face within the body bounding box
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray[y:y+h, x:x+w], scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
                    for (fx, fy, fw, fh) in faces:
                        cv2.rectangle(frame, (x + fx, y + fy), (x + fx + fw, y + fy + fh), (0, 0, 255), 2)

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for system camera, replace with the appropriate source for CCTV or other cameras

while True:
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply the background subtractor to obtain a mask
        fgmask = fgbg.apply(gray_frame)

        # Apply morphology operations to remove noise and fill gaps in the mask
        kernel = np.ones((5,5),np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the number of detected faces
        num_faces = len(contours)

        # Calculate the average brightness of the frame
        avg_brightness = np.mean(gray_frame)

        # Set the threshold value based on the number of detected faces and the average brightness
        if num_faces == 0:
            threshold_value = 30
        else:
            threshold_value = int(avg_brightness / num_faces)

        # Detect faces using the adjusted threshold value
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(threshold_value, threshold_value), flags=cv2.CASCADE_SCALE_IMAGE)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the output frame
        cv2.imshow('Face Detection', frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
'''  if not ret:
        break

    # Detect faces in the frame using MTCNN
    faces = mtcnn_detector.detect_faces(frame)

    # Define frame_color based on the number of detected faces
    num_faces = len(faces)
    if num_faces > 0:
        play_alert_sound()
        subject = 'Alert: Suspicious Activity Detected'
        message = f'The system detected {num_faces} faces in the frame. Please check the video feed.'
        sender_email = 'metaminds23@gmail.com'  # Change this to your email address
        sender_password = 'metaminds@cgu23'  # Change this to your email password
        receiver_email = 'sahoosoumya242004@gmail.com'  # Change this to the receiver's email address
        send_email(subject, message, sender_email, sender_password, receiver_email)
        frame_color = (0, 0, 255)  # Red color for the frame background
    else:
        frame_color = (0, 255, 0)  # Green color for the frame background


    for face in faces:
        x, y, width, height = face['box']
        # Draw rectangles and perform further processing (recognition, storing, etc.)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

        # Check if the detected face matches a previously detected person
        found_match = False
        for face_id, (prev_x, prev_y, prev_w, prev_h) in detected_faces.items():
            if x > prev_x and y > prev_y and x + width < prev_x + prev_w and y + height < prev_y + prev_h:
                # Detected face matches a previously detected person
                found_match = True
                # Check the human information database for details about the detected person
                info_result = check_detected_face_in_database(face_id)

                if info_result:
                    name, age, gender = info_result
                    cv2.putText(frame, f'Person {face_id}: {name}, Age: {age}, Gender: {gender}',
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, frame_color, 2)
                else:
                    cv2.putText(frame, f'Person {face_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, frame_color, 2)
                break

        if not found_match:
            # This is a new face, assign a unique ID
            detected_faces[next_unique_id] = (x, y, width, height)
            cv2.putText(frame, f'Person {next_unique_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            timestamp = datetime.now()
            try:
                cursor.execute("INSERT INTO face_detections (timestamp, x, y, width, height, unique_id) VALUES (?, ?, ?, ?, ?, ?)",
                               (timestamp, x, y, width, height, next_unique_id))
                conn.commit()
            except sqlite3.Error as e:
                print("SQLite Error:", e)
            next_unique_id += 1


    # Calculate and display frame rate
    frame_rate = calculate_frame_rate()
    cv2.putText(frame, f'Frame Rate: {frame_rate:.2f} FPS', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # Add current time and date to the frame
    add_timestamp(frame)

    # Display the frame
    cv2.imshow('Video Feed', frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the camera, close the database connection, and close all windows
cap.release()
conn.close()
cv2.destroyAllWindows()'''
