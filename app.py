import cv2
import winsound
import os
import time
from flask import Flask, render_template, request, redirect, url_for
from threading import Thread

app = Flask(__name__, template_folder="templates")

# Set up video writer
output_base = 'motion_capture_{}.avi'

fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_rate = 20.0  # Adjust the frame rate as needed
frame_size = None

# Global variables to control motion detection
cam = None
motion_detected = False
out = None
start_time = None

# Set the maximum recording duration on motion detection (in seconds)
max_record_duration = 10  # Initialize it

# Initialize frame_size here
frame_size = (640, 480)  # Change the dimensions as needed

# Function to start motion detection
@app.route('/start_detection', methods=['POST'])
def start_detection():
    global motion_detected, out, start_time, cam
    if not motion_detected:
        motion_detected = True
        start_time = time.time()
        current_time = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_base.format(current_time)
        out = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)
        
        # Open the camera
        cam = cv2.VideoCapture(0)
        
        # Start motion detection in a separate thread
        motion_thread = Thread(target=perform_motion_detection)
        motion_thread.start()
    return redirect(url_for('index'))

def perform_motion_detection():
    global motion_detected, out, start_time

    while motion_detected:
        ret, frame1 = cam.read()
        ret, frame2 = cam.read()

        # Calculate the difference between consecutive frames
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 5000:
                continue
            x, y, w, h = cv2.boundingRect(c)

            # Draw a rectangle around the moving object
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Start recording if not already recording
            if not motion_detected:
                motion_detected = True
                start_time = time.time()

                # Generate a unique filename based on the current timestamp
                current_time = time.strftime("%Y%m%d_%H%M%S")
                output_file = output_base.format(current_time)

                # Create a new video writer for each clip
                out = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

        # Add the frame to the video if recording
        if motion_detected:
            out.write(frame1)

        # Check if recording should stop after 10 seconds of inactivity
        if motion_detected and (time.time() - start_time) > max_record_duration:
            motion_detected = False
            out.release()
            winsound.PlaySound('alert.wav', winsound.SND_ASYNC)

        # Display the frame in a window
        cv2.imshow('My Cam', frame1)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(10) == ord('q'):
            break

# Function to stop motion detection
def stop_detection():
    global motion_detected
    motion_detected = False
    if out is not None:
        out.release()  # Release the video writer if it's active
    if cam is not None:
        cam.release()  # Release the camera

@app.route('/stop_detection', methods=['POST'])
def stop_detection_route():
    stop_detection() 
    return redirect(url_for('index'))

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html', motion_detected=motion_detected)

if __name__ == '__main__':
    app.run(debug=True)
