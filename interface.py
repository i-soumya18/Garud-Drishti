import sys
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QPushButton, QSizePolicy, \
    QHBoxLayout, QFrame, QGridLayout, QMessageBox
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import winsound

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class App(QDialog):
    def __init__(self):
        super().__init__()
        self.cam_id = 0  # Initialize default camera ID to 0
        self.title = 'Object and Face Detection'
        self.left = 100
        self.top = 100
        self.width = 1000
        self.height = 700
        self.detected_crowd = False  # Initialize variable for detected crowd
        self.initUI()

        # Initialize variables and functions for object and face detection
        try:
            self.conn = sqlite3.connect('detection_database.db')
            self.cursor = self.conn.cursor()
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS object_detections
                            (timestamp DATETIME, object_label TEXT, confidence REAL)''')
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS face_detections
                            (timestamp DATETIME, x INT, y INT, width INT, height INT, unique_id INT)''')
            self.conn.commit()
        except sqlite3.Error as e:
            print("SQLite Error:", e)
            self.conn.close()
            exit(1)

        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.detected_faces = {}
        self.next_unique_id = 1
        self.prev_time = datetime.now()
        self.frame_count = 0
        self.frame_rate = 0

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create outer layout
        outer_layout = QVBoxLayout()
        self.setLayout(outer_layout)

        # Create header
        header_frame = QFrame(self)
        header_frame.setMinimumHeight(80)
        header_frame.setFrameShape(QFrame.StyledPanel)

        header_layout = QHBoxLayout()
        header_frame.setLayout(header_layout)

        header_label = QLabel('Object and Face Detection', self)
        font = header_label.font()
        font.setPointSize(16)
        font.setBold(True)
        header_label.setFont(font)
        header_label.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(header_label)

        outer_layout.addWidget(header_frame)

        # Create video display
        video_frame = QFrame(self)
        video_frame.setFrameShape(QFrame.Panel)
        video_frame.setFixedSize(700, 500)

        self.image_label = QLabel(video_frame)
        self.image_label.move(0, 0)
        self.image_label.resize(video_frame.width(), video_frame.height())
        self.image_label.setScaledContents(True)

        video_layout = QHBoxLayout()
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(self.image_label)
        video_frame.setLayout(video_layout)

        outer_layout.addWidget(video_frame)

        # Create controls
        controls_frame = QFrame(self)
        controls_frame.setMinimumWidth(260)
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_layout = QVBoxLayout(controls_frame)

        self.button_start = QPushButton('Start Camera', self)
        self.button_start.clicked.connect(self.start_camera)
        self.button_start.setToolTip('Start camera')
        self.button_start.setMaximumWidth(120)

        self.button_stop = QPushButton('Stop Camera', self)
        self.button_stop.clicked.connect(self.stop_camera)
        self.button_stop.setToolTip('Stop camera')
        self.button_stop.setMaximumWidth(120)
        self.button_stop.setEnabled(False)

        self.button_change_feed = QPushButton('Change Feed', self)
        self.button_change_feed.clicked.connect(self.change_camera_feed)
        self.button_change_feed.setToolTip('Change camera feed')

        self.button_reset_database = QPushButton('Reset Database', self)
        self.button_reset_database.clicked.connect(self.reset_database)
        self.button_reset_database.setToolTip('Reset database')

        self.crowd_indicator_label = QLabel('Crowd Detection', self)
        self.crowd_indicator_label.setAlignment(Qt.AlignCenter)

        self.crowd_indicator = QLabel(self)
        self.crowd_indicator.setFixedSize(60, 60)
        self.crowd_indicator.setFrameShape(QFrame.Panel)
        self.crowd_indicator.setStyleSheet("background-color: green")

        controls_layout.addWidget(self.button_start)
        controls_layout.addWidget(self.button_stop)
        controls_layout.addWidget(self.button_change_feed)
        controls_layout.addWidget(self.button_reset_database)
        controls_layout.addWidget(self.crowd_indicator_label)
        controls_layout.addWidget(self.crowd_indicator)

        outer_layout.addWidget(controls_frame)

    def start_camera(self):
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.button_start.setEnabled(False)
        self.button_stop.setEnabled(True)
        self.button_change_feed.setEnabled(False)

    def stop_camera(self):
        self.thread.stop()
        self.button_start.setEnabled(True)
        self.button_stop.setEnabled(False)
        self.button_change_feed.setEnabled(True)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def change_camera_feed(self):
        self.cam_id = 1 - self.cam_id  # Toggle between camera IDs
        msg = QMessageBox()
        msg.setWindowTitle("Change Camera Feed")
        msg.setText(f"Switched to Camera Feed {self.cam_id}")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def reset_database(self):
        self.conn.execute("DELETE FROM object_detections")
        self.conn.execute("DELETE FROM face_detections")
        self.conn.commit()

    def calculate_frame_rate(self):
        current_time = datetime.now()
        elapsed_time = current_time - self.prev_time
        if elapsed_time.seconds >= 1:  # Update frame rate every 1 second
            self.frame_rate = self.frame_count / elapsed_time.seconds
            self.frame_count = 0
            self.prev_time = current_time

    def draw_boxes(self, frame, outs, classes):
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
                            self.cursor.execute("INSERT INTO object_detections (timestamp, object_label, confidence) VALUES (?, ?, ?)",
                                                (timestamp, label, confidence))
                            self.conn.commit()
                        except sqlite3.Error as e:
                            print("SQLite Error:", e)

                        # Check if the detected person is already in the dictionary
                        found_match = False
                        for face_id, (prev_x, prev_y, prev_w, prev_h) in self.detected_faces.items():
                            if x > prev_x and y > prev_y and x + w < prev_x + prev_w and y + h < prev_y + prev_h:
                                # Detected person matches a previously detected face
                                found_match = True
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame, f'Person {face_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                break

                        if not found_match:
                            # This is a new person, assign a unique ID
                            self.detected_faces[self.next_unique_id] = (x, y, w, h)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, f'Person {self.next_unique_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            timestamp = datetime.now()
                            try:
                                self.cursor.execute("INSERT INTO face_detections (timestamp, x, y, width, height, unique_id) VALUES (?, ?, ?, ?, ?, ?)",
                                                    (timestamp, x, y, w, h, self.next_unique_id))
                                self.conn.commit()
                            except sqlite3.Error as e:
                                print("SQLite Error:", e)
                            self.next_unique_id += 1

                        # Detect and draw face within the body bounding box
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray[y:y+h, x:x+w], scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
                        for (fx, fy, fw, fh) in faces:
                            cv2.rectangle(frame, (x + fx, y + fy), (x + fx + fw, y + fy + fh), (0, 0, 255), 2)

        # Check for crowd detection
        num_faces = len(self.detected_faces)
        if num_faces > 3 and not self.detected_crowd:
            self.detected_crowd = True
            self.crowd_indicator.setStyleSheet("background-color: red")
            winsound.PlaySound('alert.wav', winsound.SND_ASYNC | winsound.SND_LOOP)
        elif num_faces <= 3 and self.detected_crowd:
            self.detected_crowd = False
            self.crowd_indicator.setStyleSheet("background-color: green")
            winsound.PlaySound(None, winsound.SND_PURGE)

    def timerEvent(self, event):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.resize(frame, (700, 500))
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Perform object detection
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers(self.net))

        # Draw bounding boxes and faces on the frame
        self.draw_boxes(frame, outs, self.get_classes())

        # Calculate and display frame rate
        self.frame_count += 1
        self.calculate_frame_rate()
        cv2.putText(frame, f'Frame Rate: {self.frame_rate:.2f} FPS', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the frame in the GUI
        qImg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.label.setPixmap(pixmap)

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def get_classes(self):
        with open("coco.names", "r") as f:
            classes = f.read().strip().split("\n")
        return classes


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(ex.exec_())