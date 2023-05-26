# Real-Time Object Detection with OpenCV and SSD MobileNet
This repository contains a Python script for real-time object detection using the webcam feed as input. The script uses the OpenCV library (Open Source Computer Vision Library) and a pre-trained model (in this case SSD MobileNet) to recognize and label objects in real time.

# Requirements
The script requires the following:

Python 3.7 or later
OpenCV 4.0 or later
Numpy
A pretrained model and the corresponding class names file. In this case, the SSD MobileNet model is used with the COCO dataset class names.
# How to Run the Script
To run the script:

Clone this repository to your local machine.
Install the required packages.
Place your pretrained model and class names file in the same directory as the script.
Run the script: python object_detection.py
# Script Overview
The script operates as follows:

1. Initialization: The script starts by setting a threshold for object detection and another threshold for non-maximum suppression. It initializes the webcam input using cv2.VideoCapture(0).

2. Class Names Loading: It reads the coco.names file, which includes the names of object classes that the pre-trained model can recognize. This list of names will be used to label detected objects in the output.

3. Model Configuration: The script configures the SSD MobileNet model by setting the input size, scale, mean, and swapping RB.

4. Real-Time Detection: The script enters a loop where it continually reads frames from the webcam, uses the model to detect objects in each frame, and draws bounding boxes and labels around detected objects.

5. Display: The detected objects, along with their labels, are displayed in real time using cv2.imshow().

6. Exit: Pressing the q key will exit the loop, causing the script to clean up and close.

# Note
Make sure your webcam is properly connected and working. If you want to use an external webcam, you may need to change the argument in cv2.VideoCapture(). Generally, 0 is for the built-in webcam, and 1 or higher is for external webcams.