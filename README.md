# Garud-Drishti
Using existing CCTV network for crowd management using AI&amp;ML.
# Object and Face Detection System

## Overview

This is a real-time Object and Face Detection System using Python, OpenCV, YOLOv3 for object detection, and Haar Cascade for face detection. It detects and tracks objects and faces in live video streams from a camera source.

## Features

- Object detection using YOLOv3.
- Face detection using Haar Cascade.
- Real-time video feed from a camera source.
- Database integration to store detection results.
- Unique ID assignment for detected faces.
- Frame rate calculation.

## Prerequisites

Before running this project, you need to have the following:

- Python 3.x installed.
- OpenCV Python library (`cv2`) installed.
- SQLite3 for Python installed.
- YOLOv3 weights, configuration file, and coco.names file (for object detection) in the project directory.
- Haar Cascade XML file for face detection.

## Setup and Usage

1. Clone this repository to your local machine:

   ```bash
   (https://github.com/i-soumya18/Garud-Drishti/edit/main/README.md)


# Navigate to the project directory:
cd object-face-detection

## Ensure you have the necessary files and dependencies mentioned in the prerequisites section.

## Run the main script:
python main.py

### The application will open a window showing the video feed from the camera source. Detected objects and faces will be highlighted with bounding boxes, and unique IDs will be assigned to detected faces.

#Press 'q' to exit the application.

# Database
#The project uses an SQLite database (detection_database.db) to store detection results. It has two tables:

-object_detections: Stores information about detected objects.
-face_detections: Stores information about detected faces, including their unique IDs.

# Customization
You can customize this project by:

-Changing the YOLOv3 model weights and configuration files for different object detection tasks.
-Modifying the Haar Cascade XML file for different face detection tasks.
-Adjusting the confidence threshold for object detection.
-Fine-tuning detection parameters according to your requirements.

# Contributing
If you'd like to contribute to this project, please follow these steps:

-Fork the repository on GitHub.
-Clone your forked repository to your local machine.
-Create a new branch for your feature or bug fix.
-Make your changes and commit them.
-Push your changes to your forked repository.
-Submit a pull request to the original repository.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
-This project uses the YOLOv3 model for object detection. For more information about YOLO, visit: YOLO: Real-Time Object Detection.
-Face detection is based on Haar Cascade Classifiers provided by OpenCV.

## Contact
If you have any questions or suggestions, please feel free to contact the author.

<a href="https://www.youtube.com/@MetaMinds-cgu" target="_blank">
  <img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/youtube.svg" alt="MetaMinds-cgu" height="30" width="40" />

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Soumyaranjan%20Sahoo-blue?style=for-the-badge&logo=linkedin)](www.linkedin.com/in/soumya-ranjan-sahoo-b06807248/)            

