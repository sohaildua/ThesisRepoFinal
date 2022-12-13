import time
import cv2
from flask import Flask, render_template, Response
import mediapipe as mp




app1 = Flask(__name__)

@app1.route('/')
def index1():
    """Video streaming home page."""
    return render_template('index.html')

def gen1():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
                continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

    # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
    # Flip the image horizontally for a selfie-view display.
            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            key = cv2.waitKey(20)
            if key == 27:
                break
    cap.release()


@app1.route('/video_feed1')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen1(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__=="__main__":
    app1.run(debug=True)